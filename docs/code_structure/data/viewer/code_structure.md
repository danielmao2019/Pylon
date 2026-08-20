# Data Viewer Code Structure

## 1. Inheritance / type trees

`./data/viewer/utils/displays/utils/ts/backend/schemas/display_response.py`

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

`./data/viewer/utils/displays/utils/class_colors.py`

```text
class_colors.py
├── from typing import Dict, Tuple
├── import torch
├── def get_class_color(class_id: int) -> Tuple[int, int, int]
│   ├── # Maps one class identifier onto a stable palette color, wrapping the palette for ids past its end.
│   ├── impls assert isinstance(class_id, int)
│   ├── impls assert class_id >= 0
│   ├── impls palette = [(37, 99, 235), (220, 38, 38), (22, 163, 74), (202, 138, 4), (147, 51, 234), (8, 145, 178), (234, 88, 12), (79, 70, 229)]
│   └── return palette[class_id % len(palette)]
└── def map_class_ids_to_rgb(class_ids: torch.Tensor) -> Dict[int, Tuple[int, int, int]]
    ├── # Maps each distinct class id to a deterministic RGB color from a fixed class-color palette.
    ├── impls assert isinstance(class_ids, torch.Tensor)
    ├── impls assert class_ids.numel() > 0
    ├── impls flattened_class_ids = class_ids.detach().cpu().reshape(-1).to(torch.int64)
    ├── impls unique_class_ids = torch.unique(flattened_class_ids, sorted=True)
    └── return each unique class id cast to int, paired with get_class_color(class_id=that int)
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
    ├── # Base of the display wire contract: the slot, title, kind, resource url and meta_info every display response carries.
    ├── slot_id       # common field
    ├── title         # common field
    ├── display_kind  # common field
    ├── url           # common field
    └── meta_info     # common field
```

`./data/viewer/utils/displays/utils/ts/frontend/types/display_response.ts`

```text
display_response.ts
└── interface DisplayResponse
    ├── # Base of the display wire contract: the slot, title, kind, resource url and meta_info every display response carries.
    ├── slot_id       # common field
    ├── title         # common field
    ├── display_kind  # common field
    ├── url           # common field
    └── meta_info     # common field
```

`./data/viewer/utils/displays/utils/ts/backend/schemas/layered_display_response.py`

```text
layered_display_response.py
├── from typing import List, Literal
├── from data.viewer.utils.displays.utils.ts.backend.schemas.display_response import DisplayResponse
├── RASTER_DISPLAY_KINDS   # frozenset[str]: color_image, depth_image, edge_image, normal_image, segmentation_image, instance_surrogate_image, video, aabb_2d — the single source of the raster/spatial taxonomy
├── SPATIAL_DISPLAY_KINDS  # frozenset[str]: color_pc, segmentation_pc, color_gs, segmentation_gs, scene_graph, camera, aabb_3d
└── class LayeredDisplayResponse(DisplayResponse)
    ├── # Composite response stacking ordered auxiliary layers over one base layer, all resolving to a single composable class.
    ├── slot_id  # common field
    ├── title    # common field
    ├── display_kind = "layered"  # common field
    ├── url        # common field
    ├── meta_info  # common field
    ├── base_display_response: DisplayResponse        # the single base layer
    ├── aux_display_responses: List[DisplayResponse]  # ordered auxiliary layers stacked on top of the base; each consumer assigns its own per-layer semantics and owns its own visibility state
    ├── layer_class: Literal["raster", "spatial"]     # the single composable class shared by all non-placeholder layers; assigned in model_post_init and serialized so the frontend reads it instead of re-deriving the taxonomy
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
    ├── # Composite response stacking ordered auxiliary layers over one base layer, all resolving to a single composable class.
    ├── slot_id  # common field
    ├── title    # common field
    ├── display_kind: "layered"  # common field
    ├── url        # common field
    ├── meta_info  # common field
    ├── base_display_response: DisplayResponse
    ├── aux_display_responses: DisplayResponse[]
    └── layer_class: "raster" | "spatial"  # backend-stamped (layered_display_response.layer_class); the frontend reads it instead of re-deriving the raster/spatial taxonomy
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
├── import "data/viewer/utils/displays/utils/ts/frontend/register_layer_renderers";  # side-effect: eager-glob-loads every modality so its self-registration populates the registry before any render
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
│   ├── calls createTrackballCameraControls({ container, camera, renderer, initialCameraState })    → controls  # the one shared camera owns the controls
│   ├── calls _syncCameraState({ container, controls })                         # publish this cell's shared-camera pose now and on every change for cross-cell sync
│   ├── calls attachThreeScenePickSeam({ container, camera, scenes: [scene] })  # augment the container with the pickAt seam over the one shared scene
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
│   │   ├── calls reconcileInto({ root: cell, virtualTree: layerRenderer({ displayResponse: layer }) })  # mount the layer's LeafVNode into its cell
│   │   └── if layer is an aux overlay (not the base layer)
│   │       └── impls cell.style.visibility = "hidden"  # hidden until its viewBox aligns to the shared raster frustum
│   ├── impls on the base raster layer's image load (or immediately if already complete), sets each aux overlay's SVG viewBox to _alignRasterFrustum({ baseImage }) (the base image's natural extent)
│   ├── impls after setting each aux overlay's viewBox, sets that aux cell's visibility = "visible"  # revealed only once aligned to the shared raster frustum
│   └── return LeafVNode keyed by layeredDisplayResponse.slot_id whose render() returns container
├── function _syncCameraState({ container, controls }: { container: HTMLDivElement; controls: ReturnType<typeof createTrackballCameraControls> }): void
│   ├── # Publishes this cell's shared-camera pose now and re-publishes on every controls change, so other cells can observe and sync to it.
│   ├── calls _publishCameraState({ container, controls })  # initial pose
│   ├── impls controls.addEventListener("change", () => _publishCameraState({ container, controls }))  # re-publish on change
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
    └── return { width: baseImage.naturalWidth, height: baseImage.naturalHeight }  # the cell's shared frustum
```

`./data/viewer/utils/displays/utils/ts/frontend/layer_renderer_registry.ts`

```text
layer_renderer_registry.ts
├── import * as THREE from "three";
├── import type { LeafVNode } from "web/reconcile/reconcile";
├── import type { DisplayResponse } from "data/viewer/utils/displays/utils/ts/frontend/types/display_response";
├── export type SpatialLayerRenderer = ({ displayResponse }: { displayResponse: DisplayResponse }) => THREE.Object3D  # one spatial display response's part-B: build and return the THREE object the layered container adds to its shared scene
├── export type RasterLayerRenderer = ({ displayResponse }: { displayResponse: DisplayResponse }) => LeafVNode        # one raster display response's part-B: build and return the full-bleed node the layered container stacks; the container aligns the aux overlays to the shared raster frustum on the base image's load
├── const _spatialLayerRenderers = new Map<string, SpatialLayerRenderer>()  # display_kind -> spatial part-B; the module's single owner of the spatial registry, mutated only through the functions below
├── const _rasterLayerRenderers = new Map<string, RasterLayerRenderer>()    # display_kind -> raster part-B; the module's single owner of the raster registry, mutated only through the functions below
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
├── export type PickableThreeContainer = HTMLDivElement & { pickAt: (clientX: number, clientY: number) => THREE.Object3D | null }  # any spatial display container augmented with an additive base-camera pick seam: a consumer raycasts a pointer position against the container's scenes via the camera without owning the camera/renderer/scenes; the base HTMLDivElement contract is unchanged
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
│   ├── # Shared PerspectiveCamera factory for every TS atomic spatial display; the consumer-supplied initialCameraState is the single source of initial framing.
│   ├── impls THREE.PerspectiveCamera(fov=DEFAULT_TRACKBALL_PERSPECTIVE_CAMERA_FOV, ...) at default aspect/near/far/position
│   ├── if initialCameraState is not null
│   │   └── calls applyCameraStateToThreeCamera({ camera, cameraState: initialCameraState })  # so first paint matches the source display
│   └── return
├── function applyCameraStateToThreeCamera({ camera, cameraState }: { camera: THREE.PerspectiveCamera; cameraState: CameraState }): void
│   ├── # Overlays a CameraState's intrinsics and extrinsics onto a PerspectiveCamera, ignoring a malformed state.
│   ├── calls _isVectorRecord(position)
│   ├── calls _isVectorRecord(up)
│   ├── if either is not a vector record, or fov, aspect, near or far is not a number
│   │   └── return
│   ├── impls camera.position.set(position.x, position.y, position.z)
│   ├── impls camera.up.set(up.x, up.y, up.z)
│   ├── calls _isQuaternionRecord(quaternion)
│   ├── if it is a quaternion record
│   │   └── impls camera.quaternion.set(quaternion.x, quaternion.y, quaternion.z, quaternion.w)
│   ├── impls camera.fov, camera.aspect, camera.near and camera.far take the intrinsics  # impls-node-one-step:skip
│   └── impls camera.updateProjectionMatrix()
├── function _isQuaternionRecord(value: unknown): value is { x: number; y: number; z: number; w: number }
│   ├── # A quaternion record is a vector record that also carries a numeric w.
│   ├── calls _isVectorRecord(value)
│   └── return that, and whether w is a number
├── function _isVectorRecord(value: unknown): value is { x: number; y: number; z: number }
│   ├── # A vector record is a non-null object whose x, y and z are all numbers.
│   └── return that predicate
├── function createThreeWebGLRenderer({ container }: { container: HTMLDivElement }): THREE.WebGLRenderer
│   ├── # Shared WebGL renderer factory for every TS atomic spatial display.
│   ├── impls renderer = new THREE.WebGLRenderer({ alpha: true })
│   ├── impls renderer.setClearColor(0x000000, 0)  # transparent canvas by default; an opaque backdrop is the consumer's CSS background-color on the marker
│   ├── impls canvas mounted inside the provided container
│   └── return
├── function createThreeScene(): THREE.Scene
│   ├── # Shared empty-scene factory used by every TS atomic spatial display; callers scene.add their own object(s).
│   ├── impls scene = new THREE.Scene()  # the renderer's clear color is what gets visibly drawn
│   └── return
├── function attachThreeScenePickSeam({ container, camera, scenes }: { container: HTMLDivElement; camera: THREE.PerspectiveCamera; scenes: readonly THREE.Scene[] }): void
│   ├── # Installs a base-camera pickAt seam onto any spatial display container so a consumer can hit-test the given scenes via the camera without owning the camera, renderer, or scenes.
│   ├── impls raycaster = new THREE.Raycaster()
│   ├── function pickAt(clientX: number, clientY: number): THREE.Object3D | null [local]
│   │   ├── # The installed hit-test seam: maps a client point into the container's NDC and returns the first object the camera ray hits.
│   │   ├── impls rect = the container's bounding client rect
│   │   ├── if the rect is empty
│   │   │   └── return  # null: there is nothing to hit-test against
│   │   ├── impls ndc = new THREE.Vector2(((clientX - rect.left) / rect.width) * 2 - 1, -((clientY - rect.top) / rect.height) * 2 + 1)  # client space is y-down, NDC is y-up
│   │   ├── calls raycaster.setFromCamera(ndc, camera)
│   │   ├── for each scene of scenes
│   │   │   ├── impls intersections = raycaster.intersectObjects(scene.children, true)
│   │   │   └── if intersections is non-empty
│   │   │       └── return  # that first hit's object
│   │   └── return  # null: no scene was hit
│   ├── impls (container as PickableThreeContainer).pickAt = pickAt  # additive seam; base HTMLDivElement contract unchanged
│   └── return
└── function startThreeSceneRenderLoop({ scene, camera, renderer, controls, onAfterRender }: { scene: THREE.Scene; camera: THREE.PerspectiveCamera; renderer: THREE.WebGLRenderer; controls: ReturnType<typeof createTrackballCameraControls> | null; onAfterRender?: () => void }): void
    ├── # Shared runtime every spatial display runs: fits the renderer buffer, camera aspect, and trackball screen to the canvas on each resize, and drives the requestAnimationFrame loop that self-stops once the canvas leaves the DOM.
    ├── function fit(): void [local]
    │   ├── # The callback the ResizeObserver drives on every canvas resize.
    │   ├── calls renderer.setSize(renderer.domElement.clientWidth, renderer.domElement.clientHeight, false)
    │   ├── impls camera.aspect = renderer.domElement.clientWidth / renderer.domElement.clientHeight
    │   ├── calls camera.updateProjectionMatrix
    │   ├── if controls is not null
    │   │   └── calls controls.handleResize
    │   └── return
    ├── impls new ResizeObserver(fit).observe(renderer.domElement)
    ├── impls wasConnected = false  # the canvas is not appended until after render() returns, so only a later disconnect counts as an unmount
    ├── def draw
    │   ├── # The requestAnimationFrame callback: stops and frees the context once the canvas leaves the DOM, otherwise renders one frame and reschedules itself.
    │   ├── impls connected = renderer.domElement.isConnected
    │   ├── if connected
    │   │   └── impls wasConnected = true
    │   ├── if wasConnected and not connected  # canvas detached → the cell was unmounted
    │   │   ├── calls renderer.dispose
    │   │   ├── calls renderer.forceContextLoss
    │   │   └── return  # stop the loop without rescheduling
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
├── DEFAULT_POINT_SIZE_FLOOR = 0.005  # absolute floor for visibility at typical canonical-world camera framings; used by the bounding-sphere heuristic when point_size is not supplied
├── DEFAULT_POINT_SIZE_RATIO = 0.002  # fraction of point-cloud bounding-sphere radius used as the heuristic default size; lib-owned default, documented + overridable
├── DEFAULT_POINT_COLOR = "#cccccc"   # uniform fallback color used when the point cloud has no per-point colors AND the caller does not supply point_color; lib-owned default, overridable
├── def create_dash_points_display(point_cloud: PointCloud, point_size: Optional[float] = None, point_color: Optional[str] = None) -> dcc.Graph
│   ├── # Renders a Dash point-cloud display element; point_size and point_color overrides are opt-in.
│   ├── calls create_dash_points_scene(point_cloud=point_cloud, point_size=point_size, point_color=point_color)  # point_color when supplied replaces per-point colors with a uniform color, so a consumer can override the rendered look without rebuilding the data
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
    ├── impls assert isinstance(scene, go.Scatter3d)
    └── return dcc.Graph(figure=go.Figure(data=[scene]))  # the point-cloud display element
```

`./data/viewer/utils/displays/points/ts/backend/schemas/display_response.py`

```text
display_response.py
├── from data.viewer.utils.displays.utils.ts.backend.schemas.display_response import DisplayResponse
├── class PointDisplayResponse(DisplayResponse)
│   ├── # Base of the point-cloud display family: a response whose served resource is a point cloud.
│   ├── slot_id       # common field
│   ├── title         # common field
│   ├── display_kind  # common field
│   ├── url           # common field
│   └── meta_info     # common field
├── class ColorPCDisplayResponse(PointDisplayResponse)
│   ├── # Point-cloud display carrying per-point RGB color.
│   ├── slot_id  # common field
│   ├── title    # common field
│   ├── display_kind = "color_pc"  # common field
│   ├── url        # common field
│   └── meta_info  # common field
└── class SegmentationPCDisplayResponse(PointDisplayResponse)
    ├── # Point-cloud display carrying per-point class ids the backend colorizes before serving.
    ├── slot_id  # common field
    ├── title    # common field
    ├── display_kind = "segmentation_pc"  # common field
    ├── url        # common field
    └── meta_info  # common field
```

`./data/viewer/utils/displays/points/ts/backend/apis.py`

```text
apis.py
├── from pathlib import Path
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
│   ├── calls _segmentation_pc_class_ids(segmentation_pc)  # label = the returned class ids, cast to torch.int64
│   ├── impls rgb = a float32 zeros tensor of shape (segmentation_pc.num_points, 3) on the cloud's device
│   ├── for each class_id, color in class_id_to_rgb.items()
│   │   └── impls rgb[label == int(class_id)] = color  # a label with no mapping entry keeps rgb 0
│   ├── impls colorized_pc = PointCloud(xyz=segmentation_pc.xyz, data=the cloud's fields other than xyz / rgb / colors, plus rgb under the "rgb" key)
│   ├── calls _colorized_segmentation_pc_path(segmentation_pc_path=segmentation_pc_path)  # output_path
│   ├── calls save_point_cloud(pc=colorized_pc, output_filepath=str(output_path))
│   └── return str(output_path)  # the colorized point-cloud path the response serves
├── def _build_segmentation_pc_meta_info(class_id_to_rgb: Dict[int, Tuple[int, int, int]]) -> Dict[str, Any]
│   ├── # Builds factual class/color metadata from the class-to-RGB mapping.
│   ├── impls stores `class_id_to_rgb`
│   └── return
├── def _colorized_segmentation_pc_path(segmentation_pc_path: str) -> Path
│   ├── # Builds the deterministic colorized display path beside the class-labeled input resource.
│   ├── impls assert isinstance(segmentation_pc_path, str)
│   ├── impls path = Path(segmentation_pc_path)
│   └── return path.with_name("%s.viewer_colorized%s" % (path.stem, path.suffix))  # the colorized output path
└── def _segmentation_pc_class_ids(segmentation_pc: PointCloud) -> Optional[torch.Tensor]
    ├── # Returns a loaded segmentation cloud's per-point class ids from whichever field carries them.
    ├── impls assert isinstance(segmentation_pc, PointCloud)
    ├── impls field_names = segmentation_pc.field_names()
    ├── if "label" in field_names
    │   └── return segmentation_pc.label
    ├── if "feat" in field_names
    │   └── return segmentation_pc.feat
    └── return None  # the point cloud is already colorized
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
│   ├── # Base of the point-cloud display family: a response whose served resource is a point cloud.
│   ├── slot_id       # common field
│   ├── title         # common field
│   ├── display_kind  # common field
│   ├── url           # common field
│   └── meta_info     # common field
├── interface ColorPCDisplayResponse extends PointDisplayResponse
│   ├── # Point-cloud display carrying per-point RGB color.
│   ├── slot_id  # common field
│   ├── title    # common field
│   ├── display_kind = "color_pc"  # common field
│   ├── url        # common field
│   └── meta_info  # common field
└── interface SegmentationPCDisplayResponse extends PointDisplayResponse
    ├── # Point-cloud display carrying per-point class ids the backend colorizes before serving.
    ├── slot_id  # common field
    ├── title    # common field
    ├── display_kind = "segmentation_pc"  # common field
    ├── url        # common field
    └── meta_info  # common field
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
│   ├── # Renders the backend-colorized segmentation display and legend derived from meta_info; per-point colors are already baked in by the backend's class-id → rgb mapping.
│   ├── calls renderPointsDisplay({ displayResponse, initialCameraState, pointSize })
│   └── return
└── impls registerSpatialLayerRenderer({ displayKind: "color_pc", layerRenderer: createPointsObject })  # module-load self-registration of the spatial color-pc layer renderer
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
├── const DEFAULT_POINT_SIZE_FLOOR = 0.005  # number — absolute floor for visibility at typical canonical-world camera framings; used by the bounding-sphere heuristic when pointSize is not supplied
├── const DEFAULT_POINT_SIZE_RATIO = 0.002  # number — fraction of geometry bounding-sphere radius used as the heuristic default size; lib-owned default, documented + overridable
├── const DEFAULT_POINT_COLOR = "#cccccc"   # hex color — uniform fallback used when geometry has no per-point colors AND the caller does not supply pointColor; lib-owned default, overridable
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
│   ├── # Parses a PLY buffer into a BufferGeometry, dispatching on the header's declared ASCII or binary-little-endian format.
│   ├── impls headerText = the buffer's leading 1048576 bytes decoded as utf-8
│   ├── impls endIndex = headerText.indexOf("end_header")
│   ├── if endIndex < 0
│   │   └── throw new Error("PLY header is missing end_header")
│   ├── impls headerPrefix = headerText.slice(0, endIndex)
│   ├── impls dataOffset = the encoded byte length of headerText through the end of "end_header"
│   ├── while dataOffset is in range on a newline byte, 10 or 13
│   │   └── impls dataOffset += 1
│   ├── calls readPlyHeader({ headerText: headerPrefix })  → header
│   ├── if header.format === "ascii"
│   │   └── return parseAsciiPlyGeometry({ buffer, dataOffset, header })
│   ├── if header.format === "binary_little_endian"
│   │   └── return parseBinaryLittleEndianPlyGeometry({ buffer, dataOffset, header })
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
│   ├── impls material = new THREE.PointsMaterial({ vertexColors: useVertexColors, size: effectiveSize, ...(effectiveColor !== undefined ? { color: effectiveColor } : {}) })  # constructor literal is exactly these keys, and the material is used as constructed
│   └── return new THREE.Points(geometry, material)  # returned as constructed
├── function renderPointsScene({ scene, camera, renderer, controls }: { scene: THREE.Scene; camera: THREE.PerspectiveCamera; renderer: THREE.WebGLRenderer; controls: ReturnType<typeof createTrackballCameraControls>; }): void
│   ├── # Drives the point-cloud render loop with the supplied trackball controls.
│   ├── calls startThreeSceneRenderLoop({ scene, camera, renderer, controls })
│   └── return
├── function readPlyHeader({ headerText }: { headerText: string }): PlyHeader
│   ├── # Reads the vertex element's declared format, count, and scalar properties from the pre-`end_header` text.
│   ├── impls initialize the format / vertexCount / inVertex / properties accumulators the loop fills
│   ├── for each line of headerText.split(/\r?\n/)
│   │   ├── impls parts = line.trim().split(/\s+/)
│   │   ├── if parts.length === 0 || parts[0] === ""
│   │   │   └── continue
│   │   ├── if parts[0] === "format"
│   │   │   ├── impls format = parts[1]
│   │   │   └── continue
│   │   ├── if parts[0] === "element"
│   │   │   ├── impls inVertex = parts[1] === "vertex"
│   │   │   ├── if inVertex
│   │   │   │   └── impls vertexCount = Number(parts[2])
│   │   │   └── continue
│   │   └── if parts[0] === "property" && inVertex
│   │       ├── if parts[1] === "list"
│   │       │   └── throw new Error("vertex list properties are not supported")
│   │       └── impls properties.push({ type: parts[1], name: parts[2] })
│   ├── if !format
│   │   └── throw new Error("PLY format is missing")
│   ├── if !Number.isFinite(vertexCount) || vertexCount < 1
│   │   └── throw new Error(`PLY vertex count is invalid: ${vertexCount}`)
│   └── return { format, vertexCount, properties }
├── function parseAsciiPlyGeometry({ buffer, dataOffset, header }: { buffer: ArrayBuffer; dataOffset: number; header: PlyHeader }): THREE.BufferGeometry
│   ├── # Builds the geometry from the post-header rows, each row's scalars split on whitespace.
│   ├── impls lines = the post-dataOffset bytes decoded as utf-8, trimmed, split on /\r?\n/
│   ├── calls plyPropertyIndices({ properties: header.properties })  → indices
│   ├── impls positions, colors = Float32Array buffers of length header.vertexCount * 3
│   ├── for each index below header.vertexCount
│   │   ├── impls parts = lines[index]?.trim().split(/\s+/)
│   │   ├── if parts === undefined || parts.length < header.properties.length
│   │   │   └── throw new Error(`ASCII PLY row is missing vertex data: ${index}`)
│   │   ├── calls readAsciiColorComponent({ parts, index: indices.red })    → red
│   │   ├── calls readAsciiColorComponent({ parts, index: indices.green })  → green
│   │   ├── calls readAsciiColorComponent({ parts, index: indices.blue })   → blue
│   │   └── calls writeGeometryVertex({ positions, colors, index, x, y, z, red, green, blue })  # x, y, z are Number(parts[indices.x / .y / .z])
│   └── return createPointBufferGeometry({ positions, colors })
├── function parseBinaryLittleEndianPlyGeometry({ buffer, dataOffset, header }: { buffer: ArrayBuffer; dataOffset: number; header: PlyHeader }): THREE.BufferGeometry
│   ├── # Builds the geometry from the post-header bytes, read little-endian at each property's own offset and type.
│   ├── impls view = new DataView(buffer)
│   ├── calls plyPropertyOffsets({ properties: header.properties })  → offsets
│   ├── impls positions, colors = Float32Array buffers of length header.vertexCount * 3
│   ├── for each index below header.vertexCount
│   │   ├── impls base = dataOffset + index * offsets.stride
│   │   ├── calls readBinaryScalar({ view, offset: base + offsets.x.offset, type: offsets.x.type })  → x
│   │   ├── calls readBinaryScalar({ view, offset: base + offsets.y.offset, type: offsets.y.type })  → y
│   │   ├── calls readBinaryScalar({ view, offset: base + offsets.z.offset, type: offsets.z.type })  → z
│   │   ├── calls readBinaryColorComponent({ view, base, offset: offsets.red })                      → red
│   │   ├── calls readBinaryColorComponent({ view, base, offset: offsets.green })                    → green
│   │   ├── calls readBinaryColorComponent({ view, base, offset: offsets.blue })                     → blue
│   │   └── calls writeGeometryVertex({ positions, colors, index, x, y, z, red, green, blue })
│   └── return createPointBufferGeometry({ positions, colors })
├── function plyPropertyIndices({ properties }: { properties: PlyProperty[] }): PlyPropertyIndices
│   ├── # Locates each x / y / z / red / green / blue channel's column within an ASCII vertex row.
│   ├── impls names = properties.map(property => property.name)
│   ├── impls x, y, z = names.indexOf("x"), names.indexOf("y"), names.indexOf("z")
│   ├── if x < 0 || y < 0 || z < 0
│   │   └── throw new Error("PLY vertex coordinates are missing")
│   └── return { x, y, z, red: names.indexOf("red"), green: names.indexOf("green"), blue: names.indexOf("blue") }  # a channel the header omits indexes -1
├── function plyPropertyOffsets({ properties }: { properties: PlyProperty[] }): PlyPropertyOffsets
│   ├── # Locates each x / y / z / red / green / blue channel's byte offset and scalar type within one binary vertex record.
│   ├── impls offsets = {}, offset = 0
│   ├── for each property of properties
│   │   ├── impls offsets[property.name] = { offset, type: property.type }
│   │   └── calls plyScalarTypeSize({ type: property.type })  → the byte width advancing offset
│   ├── if offsets.x === undefined || offsets.y === undefined || offsets.z === undefined
│   │   └── throw new Error("PLY vertex coordinates are missing")
│   └── return { stride: offset, x: offsets.x, y: offsets.y, z: offsets.z, red: offsets.red, green: offsets.green, blue: offsets.blue }  # a channel the header omits stays undefined
├── function writeGeometryVertex({ positions, colors, index, x, y, z, red, green, blue }: { positions: Float32Array; colors: Float32Array; index: number; x: number; y: number; z: number; red: number; green: number; blue: number }): void
│   ├── # Writes one vertex's coordinates and normalized color into the two flat geometry buffers.
│   ├── impls positionOffset = index * 3
│   ├── impls positions[positionOffset], [+ 1], [+ 2] = x, y, z
│   ├── calls normalizeColorComponent({ value: red })    → colors[positionOffset]
│   ├── calls normalizeColorComponent({ value: green })  → colors[positionOffset + 1]
│   └── calls normalizeColorComponent({ value: blue })   → colors[positionOffset + 2]
├── function createPointBufferGeometry({ positions, colors }: { positions: Float32Array; colors: Float32Array }): THREE.BufferGeometry
│   ├── # Wraps the two filled buffers as a BufferGeometry with its bounding volumes computed.
│   ├── impls geometry = new THREE.BufferGeometry()
│   ├── impls geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3))  # position and color are both always set
│   ├── impls geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3))
│   ├── impls geometry.computeBoundingSphere()
│   ├── impls geometry.computeBoundingBox()
│   └── return geometry
├── function readAsciiColorComponent({ parts, index }: { parts: string[]; index: number }): number
│   ├── # Reads one color channel out of an ASCII vertex row.
│   ├── if index < 0
│   │   └── return 180  # a channel the header omits reads 180
│   └── return Number(parts[index])
├── function readBinaryColorComponent({ view, base, offset }: { view: DataView; base: number; offset: PlyPropertyOffset | undefined }): number
│   ├── # Reads one color channel out of a binary vertex record.
│   ├── if offset === undefined
│   │   └── return 180  # a channel the header omits reads 180
│   └── return readBinaryScalar({ view, offset: base + offset.offset, type: offset.type })
├── function normalizeColorComponent({ value }: { value: number }): number
│   ├── # Brings one raw color channel into the 0-1 range THREE vertex colors expect.
│   ├── if !Number.isFinite(value)
│   │   └── return 0.7
│   ├── if value <= 1
│   │   └── return Math.min(Math.max(value, 0), 1)  # already 0-1 scaled
│   └── return Math.min(Math.max(value / 255, 0), 1)  # 0-255 scaled
├── function plyScalarTypeSize({ type }: { type: string }): number
│   ├── # Gives the byte width of one declared PLY scalar type.
│   ├── impls scalarTypeSizes = the literal byte-width table over char / uchar / short / ushort / int / uint / float / double and their int8 .. float64 aliases  # impls-node-one-step:skip
│   ├── impls size = scalarTypeSizes[type]
│   ├── if size === undefined
│   │   └── throw new Error(`unsupported PLY scalar type ${type}`)
│   └── return size
└── function readBinaryScalar({ view, offset, type }: { view: DataView; offset: number; type: string }): number
    ├── # Reads one scalar of the declared PLY type little-endian out of the vertex-record DataView.
    ├── if type === "char" || type === "int8"
    │   └── return view.getInt8(offset)
    ├── if type === "uchar" || type === "uint8"
    │   └── return view.getUint8(offset)
    ├── if type === "short" || type === "int16"
    │   └── return view.getInt16(offset, true)
    ├── if type === "ushort" || type === "uint16"
    │   └── return view.getUint16(offset, true)
    ├── if type === "int" || type === "int32"
    │   └── return view.getInt32(offset, true)
    ├── if type === "uint" || type === "uint32"
    │   └── return view.getUint32(offset, true)
    ├── if type === "float" || type === "float32"
    │   └── return view.getFloat32(offset, true)
    ├── if type === "double" || type === "float64"
    │   └── return view.getFloat64(offset, true)
    └── throw new Error(`unsupported PLY scalar type ${type}`)
```

`./data/viewer/utils/displays/pixels/dash/apis.py`

```text
apis.py
├── import torch
├── from dash import dcc
├── from data.viewer.utils.displays.pixels.dash.core_pixels_display import create_dash_pixels_display
├── from data.viewer.utils.displays.utils.class_colors import map_class_ids_to_rgb
├── from data.viewer.utils.displays.utils.heatmap_colors import map_scalars_to_rgb
├── from utils.io.image import load_image
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
│   ├── # Maps the depth image to RGB through the continuous heatmap palette for Dash display.
│   ├── impls assert isinstance(depth_image_path, str)
│   ├── calls load_image(filepath=depth_image_path, normalization=None)
│   ├── if depth_image.ndim == 3
│   │   └── impls depth_image = depth_image[0]
│   ├── impls depth_scalars = depth_image.to(torch.float64)
│   ├── impls depth_scalars = depth_scalars - float(depth_scalars.min().item())
│   └── return map_scalars_to_rgb(scalars=depth_scalars)  # an HWC uint8 RGB image
├── def _map_edge_image_to_rgb
│   ├── # Maps the edge image to RGB for Dash display.
│   ├── impls assert isinstance(edge_image_path, str)
│   ├── calls load_image(filepath=edge_image_path, normalization=None)
│   ├── if edge_image.ndim == 3
│   │   └── impls edge_image = edge_image[0]
│   ├── impls edge_float = edge_image.to(torch.float64)
│   ├── impls edge_min = float(edge_float.min().item())
│   ├── impls edge_max = float(edge_float.max().item())
│   ├── impls normalized = (edge_float - edge_min) / max(edge_max - edge_min, 1e-12)
│   ├── impls gray = (normalized * 255.0).round().clamp(min=0.0, max=255.0).to(torch.uint8)
│   └── return gray.unsqueeze(-1).repeat(1, 1, 3)  # an HWC uint8 grayscale-as-RGB image
├── def _map_normal_image_to_rgb
│   ├── # Maps the normal vectors to RGB for Dash display.
│   ├── impls assert isinstance(normal_image_path, str)
│   ├── calls load_image(filepath=normal_image_path, normalization=None)
│   ├── impls normal_float = normal_image.to(torch.float64)
│   ├── impls normal_float = normal_float / 127.5 - 1.0  # decodes the stored bytes back to normal components in [-1, 1]
│   ├── impls normals_normalized = (normal_float + 1.0) / 2.0
│   ├── impls normals_normalized = normals_normalized.clamp(min=0.0, max=1.0)
│   ├── impls rgb = (normals_normalized * 255.0).round().clamp(min=0.0, max=255.0).to(torch.uint8)
│   └── return rgb.permute(1, 2, 0)  # the CHW result laid out as an HWC uint8 RGB image
├── def _map_segmentation_image_to_rgb
│   ├── # Maps the segmentation image's per-pixel class ids to RGB via the class-to-RGB mapping for Dash display.
│   ├── impls assert isinstance(segmentation_image_path, str)
│   ├── impls assert isinstance(class_id_to_rgb, dict)
│   ├── calls load_image(filepath=segmentation_image_path, normalization=None)
│   ├── impls segmentation_image = that loaded image cast to torch.int64
│   ├── impls assert segmentation_image.ndim == 2
│   ├── impls height, width = segmentation_image.shape
│   ├── impls rgb_image = a uint8 zeros tensor of shape (height, width, 3)
│   ├── for each class_id, color in class_id_to_rgb.items()
│   │   └── impls rgb_image[segmentation_image == class_id] = color  # a class id with no mapping entry stays black
│   └── return rgb_image  # an HWC uint8 RGB image
└── def _map_instance_surrogate_image_to_rgb
    ├── # Maps the instance-surrogate offset image to RGB via the class-to-RGB mapping for Dash display.
    ├── impls assert isinstance(image_path, str)
    ├── impls assert isinstance(class_id_to_rgb, dict)
    ├── calls load_image(filepath=image_path, normalization=None)
    ├── impls assert instance_surrogate is a 3-D tensor whose first dimension is at least 2
    ├── impls y_offset = instance_surrogate[0].to(torch.float64)
    ├── impls x_offset = instance_surrogate[1].to(torch.float64)
    ├── impls magnitude = torch.sqrt(y_offset**2 + x_offset**2)
    ├── impls class_id_image = torch.zeros_like(magnitude, dtype=torch.int64)
    ├── impls percentiles = torch.quantile(magnitude.reshape(-1), torch.linspace(0, 1, 20, dtype=torch.float64))
    ├── for bin_index in range(len(percentiles) - 1)
    │   ├── if bin_index == len(percentiles) - 2
    │   │   └── impls mask = magnitude >= percentiles[bin_index]
    │   ├── else
    │   │   └── impls mask = (magnitude >= percentiles[bin_index]) & (magnitude < percentiles[bin_index + 1])
    │   └── impls class_id_image[mask] = bin_index + 1
    ├── impls height, width = class_id_image.shape
    ├── impls rgb_image = a uint8 zeros tensor of shape (height, width, 3)
    ├── for each class_id, color in class_id_to_rgb.items()
    │   └── impls rgb_image[class_id_image == class_id] = color  # a class id with no mapping entry stays black
    └── return rgb_image  # an HWC uint8 RGB image
```

`./data/viewer/utils/displays/pixels/dash/core_pixels_display.py`

```text
core_pixels_display.py
├── from typing import Any
├── import numpy as np
├── import plotly.graph_objects as go
├── import torch
├── from dash import dcc
└── def create_dash_pixels_display(image: Any, image_interpolation: str) -> dcc.Graph
    ├── # Renders a Dash pixel-image display element from the resolved interpolation choice; modality-agnostic.
    ├── impls assert isinstance(image, (np.ndarray, torch.Tensor))
    ├── impls assert isinstance(image_interpolation, str)
    ├── if isinstance(image, torch.Tensor)
    │   └── impls image_array = image.detach().cpu().numpy()
    ├── else
    │   └── impls image_array = image
    ├── impls assert image_array has shape [H, W, 3]
    ├── impls zsmooth = False when image_interpolation is "nearest", "fast" otherwise  # the caller's per-modality interpolation choice
    ├── impls figure = a go.Figure over a go.Image trace of image_array carrying that zsmooth
    ├── impls hide figure's axes, letting the image fill the cell at its own aspect ratio
    └── return dcc.Graph(figure=figure)  # the pixel display element
```

`./data/viewer/utils/displays/pixels/ts/backend/schemas/display_response.py`

```text
display_response.py
├── from data.viewer.utils.displays.utils.ts.backend.schemas.display_response import DisplayResponse
├── class PixelDisplayResponse(DisplayResponse)
│   ├── # Base of the raster display family: a response whose served resource is a pixel image.
│   ├── slot_id       # common field
│   ├── title         # common field
│   ├── display_kind  # common field
│   ├── url           # common field
│   └── meta_info     # common field
├── class ColorImageDisplayResponse(PixelDisplayResponse)
│   ├── # Raster display of an RGB color image.
│   ├── slot_id  # common field
│   ├── title    # common field
│   ├── display_kind = "color_image"  # common field
│   ├── url        # common field
│   └── meta_info  # common field
├── class DepthImageDisplayResponse(PixelDisplayResponse)
│   ├── # Raster display of a depth map.
│   ├── slot_id  # common field
│   ├── title    # common field
│   ├── display_kind = "depth_image"  # common field
│   ├── url        # common field
│   └── meta_info  # common field
├── class EdgeImageDisplayResponse(PixelDisplayResponse)
│   ├── # Raster display of an edge map.
│   ├── slot_id  # common field
│   ├── title    # common field
│   ├── display_kind = "edge_image"  # common field
│   ├── url        # common field
│   └── meta_info  # common field
├── class NormalImageDisplayResponse(PixelDisplayResponse)
│   ├── # Raster display of a surface-normal map.
│   ├── slot_id  # common field
│   ├── title    # common field
│   ├── display_kind = "normal_image"  # common field
│   ├── url        # common field
│   └── meta_info  # common field
├── class SegmentationImageDisplayResponse(PixelDisplayResponse)
│   ├── # Raster display of a per-pixel class-id map.
│   ├── slot_id  # common field
│   ├── title    # common field
│   ├── display_kind = "segmentation_image"  # common field
│   ├── url        # common field
│   └── meta_info  # common field
└── class InstanceSurrogateImageDisplayResponse(PixelDisplayResponse)
    ├── # Raster display of a per-pixel instance-surrogate map.
    ├── slot_id  # common field
    ├── title    # common field
    ├── display_kind = "instance_surrogate_image"  # common field
    ├── url        # common field
    └── meta_info  # common field
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
│   ├── # Base of the raster display family: a response whose served resource is a pixel image.
│   ├── slot_id       # common field
│   ├── title         # common field
│   ├── display_kind  # common field
│   ├── url           # common field
│   └── meta_info     # common field
├── interface ColorImageDisplayResponse extends PixelDisplayResponse
│   ├── # Raster display of an RGB color image.
│   ├── slot_id  # common field
│   ├── title    # common field
│   ├── display_kind = "color_image"  # common field
│   ├── url        # common field
│   └── meta_info  # common field
├── interface DepthImageDisplayResponse extends PixelDisplayResponse
│   ├── # Raster display of a depth map.
│   ├── slot_id  # common field
│   ├── title    # common field
│   ├── display_kind = "depth_image"  # common field
│   ├── url        # common field
│   └── meta_info  # common field
├── interface EdgeImageDisplayResponse extends PixelDisplayResponse
│   ├── # Raster display of an edge map.
│   ├── slot_id  # common field
│   ├── title    # common field
│   ├── display_kind = "edge_image"  # common field
│   ├── url        # common field
│   └── meta_info  # common field
├── interface NormalImageDisplayResponse extends PixelDisplayResponse
│   ├── # Raster display of a surface-normal map.
│   ├── slot_id  # common field
│   ├── title    # common field
│   ├── display_kind = "normal_image"  # common field
│   ├── url        # common field
│   └── meta_info  # common field
├── interface SegmentationImageDisplayResponse extends PixelDisplayResponse
│   ├── # Raster display of a per-pixel class-id map.
│   ├── slot_id  # common field
│   ├── title    # common field
│   ├── display_kind = "segmentation_image"  # common field
│   ├── url        # common field
│   └── meta_info  # common field
└── interface InstanceSurrogateImageDisplayResponse extends PixelDisplayResponse
    ├── # Raster display of a per-pixel instance-surrogate map.
    ├── slot_id  # common field
    ├── title    # common field
    ├── display_kind = "instance_surrogate_image"  # common field
    ├── url        # common field
    └── meta_info  # common field
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
└── impls registerRasterLayerRenderer({ displayKind: "color_image", layerRenderer: renderColorImageDisplay })  # module-load self-registration of the raster color-image layer renderer
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
├── from dash import html
└── def create_placeholder_display
    ├── # Builds the Dash missing-result placeholder display from a message.
    ├── impls assert isinstance(message, str)
    └── return html.Div(message, className="placeholder-surface")  # the slot's stand-in element
```

`./data/viewer/utils/displays/placeholders/ts/backend/schemas/display_response.py`

```text
display_response.py
├── from data.viewer.utils.displays.utils.ts.backend.schemas.display_response import DisplayResponse
└── class PlaceholderDisplayResponse(DisplayResponse)
    ├── # Stand-in response for a slot whose artifact this selection does not have.
    ├── slot_id  # common field
    ├── title    # common field
    ├── display_kind = "placeholder"  # common field
    ├── url        # common field
    ├── meta_info  # common field
    └── message    # additional field
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
    ├── # Stand-in response for a slot whose artifact this selection does not have.
    ├── slot_id  # common field
    ├── title    # common field
    ├── display_kind = "placeholder"  # common field
    ├── url        # common field
    ├── meta_info  # common field
    └── message    # additional field
```

`./data/viewer/utils/displays/placeholders/ts/frontend/placeholder_display.ts`

```text
placeholder_display.ts
├── import type { LeafVNode } from "web/reconcile/reconcile";
├── import type { PlaceholderDisplayResponse } from "./types/display_response";
├── function renderPlaceholderDisplay({ displayResponse }: { displayResponse: PlaceholderDisplayResponse }): LeafVNode
│   ├── # Renders the missing-result placeholder UI from the response's message.
│   ├── calls _renderPlaceholderElement({ displayResponse })  # the leaf's render()
│   └── return LeafVNode keyed by displayResponse.url
└── function _renderPlaceholderElement({ displayResponse }: { displayResponse: PlaceholderDisplayResponse }): HTMLElement
    ├── # Builds the centered, italic placeholder surface carrying the response's message.
    ├── impls placeholder = a "placeholder-surface" div, flex-centered at 100% by 100%, padded 1rem,  # 888 italic
    ├── impls placeholder.textContent = displayResponse.message
    └── return placeholder
```

`./data/viewer/utils/displays/videos/dash/video_display.py`

```text
video_display.py
├── from dash import html
└── def create_video_display
    ├── # Builds the Dash video display from a video path.
    ├── impls assert src is None or isinstance(src, str)
    ├── impls assert isinstance(title, str)
    ├── if src is None
    │   └── return html.Div("Placeholder for missing video.", className="placeholder-surface")
    └── return html.Div(html.Video(src=src, controls=True, title=title))  # the slot's video element
```

`./data/viewer/utils/displays/videos/ts/backend/schemas/display_response.py`

```text
display_response.py
├── from data.viewer.utils.displays.utils.ts.backend.schemas.display_response import DisplayResponse
└── class VideoDisplayResponse(DisplayResponse)
    ├── # Display of a video resource the viewer plays in the slot.
    ├── slot_id  # common field
    ├── title    # common field
    ├── display_kind = "video"  # common field
    ├── url        # common field
    └── meta_info  # common field
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
    ├── # Display of a video resource the viewer plays in the slot.
    ├── slot_id  # common field
    ├── title    # common field
    ├── display_kind = "video"  # common field
    ├── url        # common field
    └── meta_info  # common field
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
├── from dash import html
└── def create_text_display
    ├── # Builds the Dash text display from a text string.
    ├── impls assert isinstance(text, str)
    └── return html.Pre(text, className="text-display")  # the slot's text element, whitespace preserved
```

`./data/viewer/utils/displays/texts/ts/backend/schemas/display_response.py`

```text
display_response.py
├── from data.viewer.utils.displays.utils.ts.backend.schemas.display_response import DisplayResponse
└── class TextDisplayResponse(DisplayResponse)
    ├── # Display of a text resource the viewer renders in the slot.
    ├── slot_id  # common field
    ├── title    # common field
    ├── display_kind = "text"  # common field
    ├── url        # common field
    ├── meta_info  # common field
    └── text       # additional field
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
    ├── # Display of a text resource the viewer renders in the slot.
    ├── slot_id  # common field
    ├── title    # common field
    ├── display_kind = "text"  # common field
    ├── url        # common field
    ├── meta_info  # common field
    └── text       # additional field
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
├── from dash import dash_table
└── def create_table_display
    ├── # Builds the Dash table display from tabular data.
    ├── impls assert isinstance(rows, list)
    ├── impls columns = the sorted set of field names the rows carry                    # sorting the name set is what fixes column order
    └── return dash_table.DataTable(data=rows, columns=one {name, id} spec per column)  # the slot's table element
```

`./data/viewer/utils/displays/tables/ts/backend/schemas/display_response.py`

```text
display_response.py
├── from data.viewer.utils.displays.utils.ts.backend.schemas.display_response import DisplayResponse
└── class TableDisplayResponse(DisplayResponse)
    ├── # Display of a tabular resource the viewer renders as a table.
    ├── slot_id  # common field
    ├── title    # common field
    ├── display_kind = "table"  # common field
    ├── url        # common field
    └── meta_info  # common field
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
    ├── # Display of a tabular resource the viewer renders as a table.
    ├── slot_id  # common field
    ├── title    # common field
    ├── display_kind = "table"  # common field
    ├── url        # common field
    └── meta_info  # common field
```

`./data/viewer/utils/displays/tables/ts/frontend/table_display.ts`

```text
table_display.ts
├── import type { LeafVNode } from "web/reconcile/reconcile";
├── import type { TableDisplayResponse } from "./types/display_response";
├── function renderTableDisplay({ displayResponse }: { displayResponse: TableDisplayResponse }): LeafVNode
│   ├── # Renders the complete table-display UI from the table resource URL.
│   ├── function render() [local]
│   │   ├── # The leaf's render(): the placeholder when unmaterialized, else a wrap the loader fills in.
│   │   ├── if displayResponse.url is null
│   │   │   └── return a "placeholder-surface" div reading "Placeholder for a benchmark result that is not materialized yet."
│   │   ├── impls tableWrap = a "table-wrap" div reading "Loading table"
│   │   ├── calls loadTableDisplay({ tableWrap, displayResponse })  # not awaited
│   │   └── return tableWrap
│   └── return  # { kind: "leaf", key: displayResponse.url ?? `table:${displayResponse.slot_id}`, props: {}, render }
├── async function loadTableDisplay({ tableWrap, displayResponse }: { tableWrap: HTMLDivElement; displayResponse: TableDisplayResponse }): Promise<void>
│   ├── # Fetches the table resource and renders its rows into tableWrap.
│   ├── if displayResponse.url is null
│   │   └── impls throw new Error("table display response url is null")
│   ├── impls response = await fetch(displayResponse.url)
│   ├── if the response is not ok
│   │   ├── impls tableWrap.textContent reports the HTTP status
│   │   └── return
│   ├── impls text = await response.text()
│   ├── calls readRowsFromArtifact({ text, url: displayResponse.url })  # rows
│   ├── calls renderRows({ rows })
│   └── impls tableWrap.replaceChildren(that rendered table)
├── function readRowsFromArtifact({ text, url }: { text: string; url: string }): Record<string, string>[]
│   ├── # Parses the artifact text as JSONL when the url names one, else as JSON.
│   ├── if url includes ".jsonl"
│   │   ├── impls each non-blank line is JSON.parse-d
│   │   ├── calls normalizeRow(that parsed line)
│   │   └── return those normalized rows
│   ├── impls parsed = JSON.parse(text)
│   ├── if parsed is an array
│   │   └── return parsed.map(normalizeRow)
│   ├── calls isRecord(parsed)
│   ├── if it is a record
│   │   └── return a single-element array of normalizeRow(parsed)
│   └── return []
├── function renderRows({ rows }: { rows: Record<string, string>[] }): HTMLElement
│   ├── # Builds the table element, ordering the preferred columns ahead of the discovered rest.
│   ├── impls discoveredColumns = the union of every row's keys
│   ├── impls preferredColumns = ["Question", "GT Answer", "Pred Answer", "Judgement"]
│   ├── impls columns = the preferred ones present, followed by the remaining discovered ones
│   ├── impls thead carries one th per column, each with dataset.column set
│   ├── for each row in rows
│   │   └── impls tbody gains a tr whose td per column carries row[column] ?? "" with dataset.column set, the Judgement column also given a judgement-cell class by its lowercased value  # impls-node-one-step:skip
│   └── return the table built from thead and tbody
├── function normalizeRow(value: unknown): Record<string, string>
│   ├── # Coerces one parsed entry into a flat string-valued record.
│   ├── calls isRecord(value)
│   └── return each of the record's entries stringified, or an empty record when value is not a record
└── function isRecord(value: unknown): value is Record<string, unknown>
    ├── # Narrows an unknown to a plain object record.
    └── return whether value is an object that is neither null nor an array
```

`./data/viewer/utils/displays/scene_graphs/dash/scene_graph_display.py`

```text
scene_graph_display.py
├── from dash import html
└── def create_scene_graph_display
    ├── # Builds the Dash scene-graph display from a method-agnostic graph payload.
    ├── impls assert isinstance(rows, list)
    └── return html.Pre(str(rows), className="json-preview")  # the slot's scene-graph element
```

`./data/viewer/utils/displays/scene_graphs/ts/backend/schemas/display_response.py`

```text
display_response.py
├── from data.viewer.utils.displays.utils.ts.backend.schemas.display_response import DisplayResponse
└── class SceneGraphDisplayResponse(DisplayResponse)
    ├── # Spatial display of a scene graph, whose nodes and edges the viewer draws in the 3D slot.
    ├── slot_id  # common field
    ├── title    # common field
    ├── display_kind = "scene_graph"  # common field
    ├── url        # common field; serves the scene-graph payload (no leaked encoding)
    └── meta_info  # common field
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
    ├── # Spatial display of a scene graph, whose nodes and edges the viewer draws in the 3D slot.
    ├── slot_id  # common field
    ├── title    # common field
    ├── display_kind = "scene_graph"  # common field
    ├── url        # common field; serves the scene-graph payload (no leaked encoding)
    └── meta_info  # common field
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
├── const DEFAULT_NODE_SIZE = 0.02         # number — heuristic default size for node markers when the caller does not supply nodeSize; lib-owned default, overridable
├── const DEFAULT_EDGE_COLOR = "#888888"   # hex color — neutral gray fallback for edge lines when the payload does not carry an edge color AND the caller does not supply edgeColor; lib-owned default, overridable
├── const DEFAULT_EDGE_WIDTH = 1.0         # number — line width fallback for edges when the caller does not supply edgeWidth; lib-owned default, overridable
├── const DEFAULT_LABEL_FONT_SIZE = 12     # px — font size fallback for overlay labels when the caller does not supply labelFontSize; lib-owned default, overridable
├── const DEFAULT_LABEL_COLOR = "#000000"  # hex color — text color fallback for overlay labels when the caller does not supply labelColor; lib-owned default, overridable
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
│   ├── impls create the absolutely-positioned HTML overlay container layered above the canvas (default font-size = effectiveLabelFontSize px, color = effectiveLabelColor)
│   ├── impls mount the container inside the display container
│   └── return  # the overlay container
├── async function loadSceneGraphPayload({ displayResponse }: { displayResponse: SceneGraphDisplayResponse }): Promise<SceneGraphPayload>
│   ├── # Async-loads the scene-graph payload from displayResponse.url and returns the parsed payload (node/edge positions + colors + label entries).
│   ├── if displayResponse.url === null
│   │   └── throw new Error("scene graph display response url is null")
│   ├── impls response = await fetch(displayResponse.url)
│   ├── if !response.ok
│   │   └── throw new Error(`unable to load scene graph: HTTP ${response.status}`)
│   └── return (await response.json()) as SceneGraphPayload  # cast unchecked
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
    ├── impls projects each label's world position to NDC via camera
    ├── impls converts the NDC position to overlay-pixel coordinates
    ├── impls updates each label's HTML node position (left/top), font-size = effectiveLabelFontSize px, color = effectiveLabelColor
    ├── impls culls labels behind the camera or outside the viewport
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
│   ├── # renders backend-colorized segmentation mesh display; per-element colors are already baked in by the backend's class-id → rgb mapping.
│   ├── impls reads segmentation mesh class ids from segmentation_mesh_path
│   ├── calls map_class_ids_to_rgb(class_ids=torch.unique(segmentation_mesh_class_ids))
│   ├── calls _map_segmentation_mesh_to_rgb(segmentation_mesh_path=segmentation_mesh_path, class_id_to_rgb=class_id_to_rgb)
│   └── calls create_dash_mesh_display(mesh_opacity=mesh_opacity, mesh_side=mesh_side)
├── def create_heatmap_mesh_display(heatmap_mesh_path: str, mesh_opacity: Optional[float] = None, mesh_side: Optional[str] = None) -> dcc.Graph
│   ├── # renders backend-colorized heatmap mesh display; per-element colors are already baked in by the backend's scalar → rgb mapping.
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
├── import numpy as np
├── import plotly.graph_objects as go
├── import torch
├── from dash import dcc
├── from data.viewer.utils.controls.camera.camera_controls.dash.trackball_camera_controls import create_dash_trackball_camera_controls
├── DEFAULT_MESH_COLOR = "#cccccc"  # uniform fallback color used when geometry has no texture AND has no per-vertex colors AND the caller does not supply mesh_color; lib-owned default, overridable
├── DEFAULT_MESH_OPACITY = 1.0      # opaque default applied when the caller does not supply mesh_opacity; lib-owned default, overridable
├── DEFAULT_MESH_SIDE = "double"    # fallback side mode for visibility under arbitrary camera framings when the caller does not supply mesh_side; lib-owned default, overridable
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
│   ├── elif mesh.texture carries per-vertex color
│   │   ├── calls _normalize_rgb_tensor_to_uint8(rgb_values=mesh.texture.vertex_color)  # vertex_colors
│   │   ├── for each vertex_rgb row of vertex_colors
│   │   │   └── calls _rgb_to_css_color(rgb_values=vertex_rgb)
│   │   └── impls effective_color = those CSS color strings
│   ├── else
│   │   └── impls effective_color = DEFAULT_MESH_COLOR
│   └── return
├── def _create_dash_uv_texture_map_mesh_scene(mesh: Any, mesh_color: Optional[str], effective_opacity: float, effective_side: str) -> go.Mesh3d
│   ├── # Builds the Plotly Mesh3d trace for a UV-texture-mapped mesh, resolving the effective color.
│   ├── if mesh_color is not None
│   │   └── impls effective_color = mesh_color
│   ├── elif mesh.texture carries a uv_texture_map
│   │   ├── calls _normalize_texture_map_to_uint8(texture_map=mesh.texture.uv_texture_map)  # texture_map
│   │   ├── impls sampled_rgb = texture_map sampled at mesh.texture.verts_uvs
│   │   ├── for each vertex_rgb row of sampled_rgb
│   │   │   └── calls _rgb_to_css_color(rgb_values=vertex_rgb)
│   │   └── impls effective_color = those CSS color strings
│   ├── else
│   │   └── impls effective_color = DEFAULT_MESH_COLOR
│   └── return
├── def create_dash_mesh_component
│   ├── # Assembles the Dash component that hosts the Mesh3d scene and its trackball camera controls.
│   ├── impls assert isinstance(scene, go.Mesh3d)
│   └── return dcc.Graph(figure=go.Figure(data=[scene]))  # the mesh display element
├── def _normalize_rgb_tensor_to_uint8(rgb_values: torch.Tensor) -> np.ndarray
│   ├── # Normalizes one RGB tensor to uint8 numpy layout.
│   ├── if rgb_values.dtype is torch.uint8
│   │   └── return its own numpy array
│   ├── impls rgb_values.dtype is torch.float32 here
│   ├── impls the [0, 1] values scale by 255 into uint8
│   └── return  # (N, 3) uint8 numpy array
├── def _normalize_texture_map_to_uint8(texture_map: torch.Tensor) -> np.ndarray
│   ├── # Normalizes one UV texture map to uint8 numpy layout.
│   ├── if texture_map.dtype is torch.uint8
│   │   └── return its own numpy array
│   ├── impls texture_map.dtype is torch.float32 here
│   ├── impls the [0, 1] values scale by 255 into uint8
│   └── return  # (H, W, 3) uint8 numpy array
└── def _rgb_to_css_color(rgb_values: np.ndarray) -> str
    ├── # Converts one RGB triplet to a CSS color string.
    ├── impls rgb_uint8 = rgb_values clipped to [0, 255], as uint8
    └── return f"rgb({int(rgb_uint8[0])},{int(rgb_uint8[1])},{int(rgb_uint8[2])})"
```

`./data/viewer/utils/displays/mesh/ts/backend/schemas/display_response.py`

```text
display_response.py
├── from data.viewer.utils.displays.utils.ts.backend.schemas.display_response import DisplayResponse
├── class MeshDisplayResponse(DisplayResponse)
│   ├── # Base of the mesh display family: a response whose served resource is a triangle mesh.
│   ├── slot_id       # common field
│   ├── title         # common field
│   ├── display_kind  # common field
│   ├── url           # common field
│   └── meta_info     # common field
├── class ColorMeshDisplayResponse(MeshDisplayResponse)
│   ├── # Mesh display carrying per-element RGB color.
│   ├── slot_id  # common field
│   ├── title    # common field
│   ├── display_kind = "color_mesh"  # common field
│   ├── url        # common field
│   └── meta_info  # common field
├── class SegmentationMeshDisplayResponse(MeshDisplayResponse)
│   ├── # Mesh display carrying per-element class ids the backend colorizes before serving.
│   ├── slot_id  # common field
│   ├── title    # common field
│   ├── display_kind = "segmentation_mesh"  # common field
│   ├── url        # common field — the class-colorized mesh resource
│   └── meta_info  # common field
├── class HeatmapMeshDisplayResponse(MeshDisplayResponse)
│   ├── # Mesh display carrying per-element scalars the backend colorizes into a heatmap before serving.
│   ├── slot_id  # common field
│   ├── title    # common field
│   ├── display_kind = "heatmap_mesh"  # common field
│   ├── url        # common field — the heatmap-colorized mesh resource
│   └── meta_info  # common field
└── class SparseHeatmapMeshDisplayResponse(MeshDisplayResponse)
    ├── # Mesh display carrying a sparse (indices, values) heatmap delta against a referenced geometry.
    ├── slot_id  # common field
    ├── title    # common field
    ├── display_kind = "sparse_heatmap_mesh"  # common field
    ├── url        # common field — the sparse heatmap wire resource: a shared-geometry reference plus the sparse (indices, values) delta
    └── meta_info  # common field
```

`./data/viewer/utils/displays/mesh/ts/backend/apis.py`

```text
apis.py
├── import json
├── from pathlib import Path
├── from typing import Any, Dict, Tuple
├── import torch
├── from data.structures.three_d.mesh.load import load_mesh
├── from data.structures.three_d.mesh.mesh import Mesh
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
│   ├── calls _read_segmentation_mesh_class_ids(input_path=input_path)  # segmentation_mesh_class_ids
│   ├── calls map_class_ids_to_rgb(class_ids=torch.unique(segmentation_mesh_class_ids))
│   ├── calls _map_segmentation_mesh_to_rgb(input_path=input_path, output_path=output_path, class_id_to_rgb=class_id_to_rgb)
│   ├── calls _build_segmentation_mesh_meta_info(class_id_to_rgb=class_id_to_rgb)
│   ├── calls create_mesh_display_response_core
│   └── return
├── def create_heatmap_mesh_display_response(input_path: Path, output_path: Path, url: str, slot_id: str, title: str, meta_info: Dict[str, Any]) -> HeatmapMeshDisplayResponse
│   ├── # Creates a heatmap mesh response from a non-negative-scalar-labeled mesh resource read from input_path; processed mesh is written to output_path.
│   ├── calls _read_heatmap_mesh_scalars(input_path=input_path)  # heatmap_mesh_scalars
│   ├── calls map_scalars_to_rgb(scalars=heatmap_mesh_scalars)
│   ├── calls _map_heatmap_mesh_to_rgb(input_path=input_path, output_path=output_path, scalar_rgb=scalar_rgb)
│   ├── calls _build_heatmap_mesh_meta_info(scalars=heatmap_mesh_scalars)
│   ├── calls create_mesh_display_response_core
│   └── return
├── def create_sparse_heatmap_mesh_display_response(input_path: Path, output_path: Path, url: str, slot_id: str, title: str, meta_info: Dict[str, Any]) -> SparseHeatmapMeshDisplayResponse
│   ├── # Creates a sparse heatmap mesh response; writes the sparse (indices, values) delta resource to output_path.
│   ├── impls reads the (indices, values) delta and the geometry reference from input_path  # impls-node-one-step:skip
│   ├── calls _write_sparse_heatmap_resource(input_path=input_path, output_path=output_path)
│   ├── calls _build_sparse_heatmap_mesh_meta_info(indices=indices, values=values)
│   └── return SparseHeatmapMeshDisplayResponse with slot_id, title, url, meta_info from caller-provided args
├── def _map_segmentation_mesh_to_rgb(input_path: Path, output_path: Path, class_id_to_rgb: Dict[int, Tuple[int, int, int]]) -> None
│   ├── # Reads segmentation mesh from input_path, applies class_id_to_rgb, writes the resulting color mesh to output_path.
│   ├── if class-id storage is per-vertex
│   │   ├── calls _segmentation_mesh_per_vertex_class_ids(mesh=mesh)  # class_ids
│   │   └── impls assigns class_id_to_rgb[c] as the per-vertex RGB for class id c
│   ├── elif class-id storage is per-texel
│   │   ├── calls _segmentation_mesh_per_texel_class_ids(mesh=mesh)  # class_ids
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
│   ├── # Writes the (indices, values) delta + geometry_url from input_path to output_path as the wire resource.
│   ├── impls assert isinstance(input_path, Path)
│   ├── impls assert isinstance(output_path, Path)
│   ├── calls _read_sparse_heatmap_geometry_url(input_path=input_path)  # geometry_url
│   ├── calls _read_sparse_heatmap_arrays(input_path=input_path)        # indices, values
│   ├── impls payload = {"geometry_url": geometry_url, "indices": indices.tolist(), "values": values.tolist()}
│   ├── impls output_path.parent.mkdir(parents=True, exist_ok=True)
│   └── with output_path.open("w") as fh
│       └── impls json.dump(payload, fh)
├── def _build_segmentation_mesh_meta_info
│   ├── # Builds class/color metadata from the class-to-RGB mapping.
│   ├── impls stores `class_id_to_rgb`
│   └── return
├── def _build_heatmap_mesh_meta_info
│   ├── # Builds scalar-range metadata from the input scalars.
│   ├── impls stores scalar min/max
│   └── return
├── def _build_sparse_heatmap_mesh_meta_info
│   ├── # Builds scalar-range + non-zero-count metadata from the input sparse arrays.
│   ├── impls stores values min/max and number of non-zero entries  # impls-node-one-step:skip
│   └── return
├── def _read_sparse_heatmap_geometry_url(input_path: Path) -> str
│   ├── # Reads the url of the mesh resource whose vertex domain the sparse delta indexes into.
│   ├── impls assert isinstance(input_path, Path)
│   ├── with input_path.open("r") as fh
│   │   └── impls payload = json.load(fh)
│   ├── impls assert "geometry_url" in payload
│   ├── impls geometry_url = payload["geometry_url"]
│   ├── impls assert isinstance(geometry_url, str) and len(geometry_url) > 0  # impls-node-one-step:skip
│   └── return geometry_url  # the shared base mesh's resource url
├── def _read_sparse_heatmap_arrays(input_path: Path) -> Tuple[torch.Tensor, torch.Tensor]
│   ├── # Reads the non-default (indices, values) entries the sparse delta carries.
│   ├── impls assert isinstance(input_path, Path)
│   ├── with input_path.open("r") as fh
│   │   └── impls payload = json.load(fh)
│   ├── impls indices = torch.tensor(payload["indices"], dtype=torch.int64)
│   ├── impls values = torch.tensor(payload["values"], dtype=torch.float32)
│   ├── impls assert indices.ndim == 1
│   ├── impls assert values.ndim == 1
│   ├── impls assert indices.shape[0] == values.shape[0]
│   ├── impls assert bool((indices >= 0).all())
│   ├── impls assert bool((values >= 0).all())
│   └── return indices, values  # 1-D int64 vertex ids and their 1-D float32 non-negative scalars
├── def _read_segmentation_mesh_class_ids(input_path: Path) -> torch.Tensor
│   ├── # Reads per-vertex or per-texel class ids from a segmentation mesh.
│   ├── calls load_mesh(path=input_path)  # the mesh fields Mesh is built from
│   ├── if mesh.texture_mode == "vertex_color"
│   │   ├── calls _segmentation_mesh_per_vertex_class_ids(mesh=mesh)
│   │   └── return  # int64 [V] per-vertex class ids
│   ├── if mesh.texture_mode == "uv_texture_map"
│   │   ├── calls _segmentation_mesh_per_texel_class_ids(mesh=mesh)
│   │   └── return  # int64 [H, W] per-texel class ids
│   └── raise ValueError  # unsupported segmentation mesh texture_mode
├── def _read_heatmap_mesh_scalars(input_path: Path) -> torch.Tensor
│   ├── # Reads per-vertex or per-texel non-negative scalars from a heatmap mesh.
│   ├── calls load_mesh(path=input_path)  # the mesh fields Mesh is built from
│   ├── if mesh.texture_mode == "vertex_color"
│   │   └── return mesh.vertex_color[:, 0]  # non-negative [V]
│   ├── if mesh.texture_mode == "uv_texture_map"
│   │   └── return mesh.uv_texture_map[..., 0]  # non-negative [H, W]
│   └── raise ValueError  # unsupported heatmap mesh texture_mode
├── def _segmentation_mesh_per_vertex_class_ids(mesh: Mesh) -> torch.Tensor
│   ├── # Extracts per-vertex class ids from a vertex-colored segmentation mesh.
│   └── return mesh.vertex_color[:, 0].to(dtype=torch.int64)  # int64 [V]
└── def _segmentation_mesh_per_texel_class_ids(mesh: Mesh) -> torch.Tensor
    ├── # Extracts per-texel class ids from a UV-textured segmentation mesh.
    └── return mesh.uv_texture_map[..., 0].to(dtype=torch.int64)  # int64 [H, W]
```

`./data/viewer/utils/displays/mesh/ts/backend/core_mesh_display.py`

```text
core_mesh_display.py
├── from pathlib import Path
├── from typing import Any, Dict
├── from data.structures.three_d.mesh.mesh import Mesh
├── from data.structures.three_d.mesh.save import save_mesh
├── from data.structures.three_d.mesh.texture.mesh_texture_uv_texture_map import MeshTextureUVTextureMap
├── from data.structures.three_d.mesh.texture.mesh_texture_vertex_color import MeshTextureVertexColor
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
├── def _create_vertex_color_mesh_display_response(mesh: Mesh, output_path: Path) -> None
│   ├── # Writes the per-vertex-colored mesh resource to output_path.
│   ├── impls assert isinstance(mesh, Mesh)
│   ├── impls assert isinstance(output_path, Path)
│   ├── impls assert isinstance(mesh.texture, MeshTextureVertexColor)
│   └── calls save_mesh(mesh=mesh, output_path=output_path)
└── def _create_uv_texture_map_mesh_display_response(mesh: Mesh, output_path: Path) -> None
    ├── # Writes the UV-texture-mapped mesh resource to output_path.
    ├── impls assert isinstance(mesh, Mesh)
    ├── impls assert isinstance(output_path, Path)
    ├── impls assert isinstance(mesh.texture, MeshTextureUVTextureMap)
    └── calls save_mesh(mesh=mesh, output_path=output_path)
```

`./data/viewer/utils/displays/mesh/ts/frontend/types/display_response.ts`

```text
display_response.ts
├── import type { DisplayResponse } from "data/viewer/utils/displays/utils/ts/frontend/types/display_response";
├── interface MeshDisplayResponse extends DisplayResponse
│   ├── # Base of the mesh display family: a response whose served resource is a triangle mesh.
│   ├── slot_id       # common field
│   ├── title         # common field
│   ├── display_kind  # common field
│   ├── url           # common field
│   └── meta_info     # common field
├── interface ColorMeshDisplayResponse extends MeshDisplayResponse
│   ├── # Mesh display carrying per-element RGB color.
│   ├── slot_id  # common field
│   ├── title    # common field
│   ├── display_kind = "color_mesh"  # common field
│   ├── url        # common field
│   └── meta_info  # common field
├── interface SegmentationMeshDisplayResponse extends MeshDisplayResponse
│   ├── # Mesh display carrying per-element class ids the backend colorizes before serving.
│   ├── slot_id  # common field
│   ├── title    # common field
│   ├── display_kind = "segmentation_mesh"  # common field
│   ├── url        # common field — the class-colorized mesh resource
│   └── meta_info  # common field
├── interface HeatmapMeshDisplayResponse extends MeshDisplayResponse
│   ├── # Mesh display carrying per-element scalars the backend colorizes into a heatmap before serving.
│   ├── slot_id  # common field
│   ├── title    # common field
│   ├── display_kind = "heatmap_mesh"  # common field
│   ├── url        # common field — the heatmap-colorized mesh resource
│   └── meta_info  # common field
└── interface SparseHeatmapMeshDisplayResponse extends MeshDisplayResponse
    ├── # Mesh display carrying a sparse (indices, values) heatmap delta against a referenced geometry.
    ├── slot_id  # common field
    ├── title    # common field
    ├── display_kind = "sparse_heatmap_mesh"  # common field
    ├── url        # common field — the sparse heatmap wire resource: a shared-geometry reference plus the sparse (indices, values) delta
    └── meta_info  # common field
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
├── const DEFAULT_MESH_COLOR = "#cccccc"        # hex color — uniform fallback used when geometry has no texture AND has no vertex colors AND the caller does not supply meshColor; lib-owned default, overridable
├── const DEFAULT_MESH_OPACITY = 1.0            # number — opaque default applied when the caller does not supply meshOpacity; material's `transparent` flag flips true automatically when opacity is less than 1; lib-owned default, overridable
├── const DEFAULT_MESH_SIDE = THREE.DoubleSide  # THREE.Side — fallback side mode for visibility under arbitrary camera framings when the caller does not supply meshSide; lib-owned default, overridable
├── interface MeshPayload
│   ├── # The render-side mirror of the Mesh data structure: geometry (verts, faces) plus an optional MeshTexture.
│   ├── verts: Float32Array  # [V, 3] flattened — mirrors Mesh.verts
│   ├── faces: Uint32Array   # [F, 3] flattened — mirrors Mesh.faces
│   └── texture: MeshTextureVertexColor | MeshTextureUVTextureMap | null  # mirrors Mesh.texture (Optional[MeshTexture])
├── interface MeshTextureVertexColor
│   ├── # Render mirror of the data structure's MeshTextureVertexColor: per-vertex colors aligned 1:1 with verts.
│   ├── kind: "vertex_color"
│   └── vertexColor: Float32Array  # [V, C] per-vertex colors, C in {3, 4}
├── interface MeshTextureUVTextureMap
│   ├── # Render mirror of the data structure's MeshTextureUVTextureMap: a per-face-indexed UV texture map.
│   ├── kind: "uv_texture_map"
│   ├── uvTextureMap: THREE.Texture  # the texture image
│   ├── vertsUvs: Float32Array       # [VT, 2] UV coordinates
│   └── facesUvs: Uint32Array        # [F, 3] flattened — per-face UV-vertex indices
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
│   ├── impls group = new THREE.Group()
│   ├── impls loadMeshPayload({ displayResponse }).then(payload => group.add(createThreeMesh({ payload, displayResponse, meshColor, meshOpacity, meshSide })))
│   └── return group
├── async function loadMeshPayload({ displayResponse }: { displayResponse: MeshDisplayResponse }): Promise<MeshPayload>
│   ├── # Async-loads the mesh payload from displayResponse.url; resolves a sparse-heatmap delta against its referenced geometry, otherwise reads the dense resource as-is.
│   ├── if displayResponse.url is null
│   │   └── impls throw new Error("mesh display response url is null")
│   ├── if displayResponse.display_kind is "sparse_heatmap_mesh"
│   │   ├── calls _fetchSparseHeatmapResource(displayResponse.url)  # sparse
│   │   ├── calls _fetchObj(sparse.geometryUrl)                     # parsed
│   │   └── return _resolveSparseHeatmapPayload({ parsed, sparse })
│   ├── calls _fetchObj(displayResponse.url)                                    # parsed
│   ├── calls _resolveMeshTexture({ parsed, primaryUrl: displayResponse.url })  # texture
│   └── return  # { verts, faces, texture }
├── function createThreeMesh({ payload, displayResponse, meshColor, meshOpacity, meshSide }: { payload: MeshPayload; displayResponse: MeshDisplayResponse; meshColor?: string; meshOpacity?: number; meshSide?: THREE.Side }): THREE.Mesh
│   ├── # Sync-builds THREE.BufferGeometry + THREE.MeshBasicMaterial + THREE.Mesh from a pre-loaded payload.
│   ├── impls geometry = non-indexed THREE.BufferGeometry whose position attribute gathers payload.verts by payload.faces (each of the F faces contributes its 3 corner positions), so render corner c maps to logical vertex payload.faces[c]
│   ├── impls set geometry.userData.cornerVertexIndices = payload.faces  # payload.faces flattened IS this non-indexed geometry's corner→vertex map, so a downstream consumer can gather a per-logical-vertex field into the corner render domain
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
│   ├── impls material = MeshBasicMaterial { vertexColors: useVertexColors, side: effectiveSide, opacity: effectiveOpacity, transparent when opacity<1 or RGBA vertex colors, map: payload.texture.uvTextureMap when useTexture, color: effectiveColor when set }  # RGBA alpha-0 corners render transparent
│   └── return new THREE.Mesh(geometry, material)  # returned as constructed
├── function renderMeshScene({ scene, camera, renderer, controls }: { scene: THREE.Scene; camera: THREE.PerspectiveCamera; renderer: THREE.WebGLRenderer; controls: ReturnType<typeof createTrackballCameraControls>; }): void
│   ├── # Drives the mesh render loop with the supplied trackball controls.
│   ├── calls startThreeSceneRenderLoop({ scene, camera, renderer, controls })
│   └── return
├── async function _fetchSparseHeatmapResource(url: string): Promise<SparseHeatmapResource>
│   ├── # Fetches and validates a sparse-heatmap JSON resource into typed arrays.
│   ├── impls response = await fetch(url)
│   ├── if the response is not ok
│   │   └── impls throw new Error(`GET ${url} failed: ${response.status}`)
│   ├── if geometry_url is not a non-empty string
│   │   └── impls throw new Error reporting the missing geometry_url
│   ├── if indices or values is not an array
│   │   └── impls throw new Error reporting the missing indices/values arrays
│   └── return  # { geometryUrl, indices as Int32Array, values as Float32Array }
├── function _resolveSparseHeatmapPayload({ parsed, sparse }: { parsed: ParsedObj; sparse: SparseHeatmapResource }): MeshPayload
│   ├── # Paints the sparse scalars onto a per-vertex RGBA buffer, leaving unlisted vertices transparent.
│   ├── impls vertexColor = a Float32Array of 4 channels per parsed vertex
│   ├── calls _mapScalarsToRgb(sparse.values)  # rgb
│   ├── for each sparse index and its position i
│   │   ├── if the vertex index falls outside the parsed vertex range
│   │   │   └── impls throw new Error reporting the out-of-range vertex index
│   │   └── impls that vertex takes rgb[i] scaled to [0, 1] at alpha 1, every other vertex staying at alpha 0 (a base-revealing overlay)
│   └── return  # { verts, faces, texture: { kind: "vertex_color", vertexColor } }
├── function _mapScalarsToRgb(values: Float32Array): Uint8Array
│   ├── # Maps non-negative scalars through the heatmap palette, normalized by the largest value.
│   ├── impls maxValue = 0.0
│   ├── for each value
│   │   └── if it exceeds maxValue
│   │       └── impls maxValue = that value
│   ├── impls denom = Math.max(maxValue, 1e-12)
│   ├── impls rgb = a Uint8Array of three bytes per value
│   ├── for each value
│   │   ├── impls normalized = the value over denom, clamped to [0, 1]
│   │   ├── impls segment advances while normalized reaches past the next palette stop
│   │   ├── impls fraction = normalized's position within that stop pair, clamped to [0, 1]
│   │   └── impls its three rgb bytes take the rounded interpolation between the pair's colors
│   └── return rgb
├── function _fetchObj(url: string): Promise<ParsedObj>
│   ├── # Fetches and parses an OBJ, memoizing the in-flight promise per url.
│   ├── if _objCache already holds url
│   │   └── return that cached promise
│   ├── impls promise = fetch(url).then(async response => { if (!response.ok) throw new Error(`GET ${url} failed: ${response.status}`); return _parseObj(await response.text()) })
│   ├── impls _objCache.set(url, promise)
│   └── return promise
├── function _parseObj(text: string): ParsedObj
│   ├── # Parses OBJ v / vt / f / mtllib records into flat vertex, uv and face buffers.
│   ├── for each line of text
│   │   ├── if the line is empty
│   │   │   └── impls continue
│   │   ├── if the line is a "v " record
│   │   │   ├── impls its xyz joins vPositions
│   │   │   ├── if the record carries 7 or more fields
│   │   │   │   ├── impls sawVertexColors = true
│   │   │   │   └── impls its rgb joins vColors
│   │   │   └── else
│   │   │       └── impls vColors takes NEUTRAL_GRAY three times
│   │   ├── else if the line is a "vt" record
│   │   │   └── impls its uv pair joins vtCoords
│   │   ├── else if the line is an "f " record
│   │   │   ├── calls _parseFaceCorner(each corner token)  # corners
│   │   │   └── for each triangle of the corner fan (corners[0], corners[j], corners[j + 1])
│   │   │       ├── impls its three vertex indices push onto faceVertexTokens
│   │   │       ├── impls its three uv indices push onto faceUvTokens
│   │   │       └── if a corner carries a uv index
│   │   │           └── impls sawAnyUv = true
│   │   └── else if the line is an "mtllib" record
│   │       └── if it splits into 2 or more fields
│   │           └── impls mtllibName takes the rest of that line
│   ├── impls geometryVertexCount = vPositions.length / 3, cornerCount = faceVertexTokens.length
│   ├── impls useUvs = sawAnyUv over a non-empty vtCoords
│   ├── impls verts = a Float32Array over vPositions, faces = a Uint32Array of cornerCount
│   ├── impls vertexColor = a Float32Array over vColors when the OBJ carried colors, else null
│   ├── impls facesUvs, vertsUvs = a Uint32Array of cornerCount, a Float32Array over vtCoords when useUvs, else null
│   ├── for each face corner
│   │   ├── if its vertex index falls outside the geometry vertex range
│   │   │   └── impls throw new Error reporting the out-of-range vertex index
│   │   ├── impls faces[corner] = that vertex index
│   │   └── if facesUvs is not null
│   │       ├── if its uv index falls outside vtCoords
│   │       │   └── impls throw new Error reporting the missing UV index
│   │       └── impls facesUvs[corner] = that uv index
│   └── return  # verts, faces, vertexColor when the OBJ carried colors, vertsUvs/facesUvs when it carried uvs, and mtllibName
├── function _parseFaceCorner(token: string): { v: number; vt: number }
│   ├── # Splits one "v/vt" face corner into zero-based vertex and uv indices.
│   ├── impls v = the first field, converted from OBJ's 1-based index
│   └── return  # { v, vt }, with vt at -1 when the corner carries none
├── async function _resolveMeshTexture({ parsed, primaryUrl }: { parsed: ParsedObj; primaryUrl: string }): Promise<MeshTextureVertexColor | MeshTextureUVTextureMap | null>
│   ├── # Resolves the OBJ's texture: a UV map via its MTL when it has one, else per-vertex color, else none.
│   ├── if parsed carries vertsUvs, facesUvs and an mtllibName
│   │   ├── calls _siblingUrl(primaryUrl, parsed.mtllibName)
│   │   ├── calls _fetchMtlTextureName(that mtl url)  # textureName
│   │   ├── if textureName is null
│   │   │   └── impls throw new Error reporting UVs declared with no map_Kd
│   │   ├── calls _fetchTexture(_siblingUrl(primaryUrl, textureName))  # uvTextureMap
│   │   └── return  # { kind: "uv_texture_map", uvTextureMap, vertsUvs, facesUvs }
│   ├── if parsed.vertexColor is not null
│   │   └── return  # { kind: "vertex_color", vertexColor }
│   └── return null
├── async function _fetchMtlTextureName(mtlUrl: string): Promise<string | null>
│   ├── # Reads the MTL's map_Kd texture name, or null when it declares none.
│   ├── impls response = await fetch(mtlUrl)
│   ├── if the response is not ok
│   │   └── impls throw new Error(`GET ${mtlUrl} failed: ${response.status}`)
│   ├── for each line of the MTL text
│   │   └── if it starts with "map_Kd"
│   │       └── if it splits into 2 or more fields
│   │           └── return the rest of that line as the texture name
│   └── return null
├── function _siblingUrl(primaryUrl: string, siblingName: string): string
│   ├── # Resolves a sibling file name against the primary url's directory.
│   ├── if primaryUrl has no "/"
│   │   └── return siblingName
│   └── return primaryUrl up to its last "/" joined with siblingName
└── function _fetchTexture(textureUrl: string): Promise<THREE.Texture>
    ├── # Loads a THREE texture, memoizing the in-flight promise per url.
    ├── if _textureCache already holds textureUrl
    │   └── return that cached promise
    ├── impls loader = a new THREE.TextureLoader()
    ├── impls promise = a new Promise around loader.load(textureUrl)
    ├── impls its onLoad = texture => { texture.colorSpace = THREE.SRGBColorSpace; texture.flipY = true; texture.needsUpdate = true; resolve(texture) }
    ├── impls the onError callback rejects with `unable to load texture image: ${textureUrl}`
    ├── impls _textureCache.set(textureUrl, promise)
    └── return promise
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
│   ├── # renders backend-colorized mesh display and legend derived from meta_info; per-element colors are already baked in by the backend's class-id → rgb mapping.
│   ├── calls renderMeshDisplay({ displayResponse, initialCameraState, meshOpacity, meshSide })
│   └── return
├── function renderHeatmapMeshDisplay({ displayResponse, initialCameraState, meshOpacity, meshSide }: { displayResponse: HeatmapMeshDisplayResponse; initialCameraState?: CameraState | null; meshOpacity?: number; meshSide?: THREE.Side }): LeafVNode
│   ├── # renders backend-colorized mesh display and continuous-palette legend derived from meta_info (scalar min/max); per-element colors are already baked in by the backend's scalar → rgb mapping.
│   ├── calls renderMeshDisplay({ displayResponse, initialCameraState, meshOpacity, meshSide })
│   └── return
└── function renderSparseHeatmapMeshDisplay({ displayResponse, initialCameraState, meshOpacity, meshSide }: { displayResponse: SparseHeatmapMeshDisplayResponse; initialCameraState?: CameraState | null; meshOpacity?: number; meshSide?: THREE.Side }): LeafVNode
    ├── # renders the sparse heatmap mesh display and continuous-palette legend from meta_info (scalar min/max); per-element colors are already baked in by the backend's scalar → rgb mapping.
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
    ├── # Recolors the segmentation Gaussian's per-Gaussian class ids to RGB via the class-to-RGB mapping.
    ├── impls assert isinstance(segmentation_gs_path, str)
    ├── impls assert isinstance(class_id_to_rgb, dict)
    └── raise NotImplementedError("Dash segmentation-to-color Gaussian mapping is declared by the skeleton but not exercised by any caller in this branch.")
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
│   ├── impls Dash Gaussian-splat display scene from Gaussian data and display metadata  # impls-node-one-step:skip
│   └── return
└── def create_dash_gaussians_component
    ├── # Assembles the Dash component that hosts the Gaussian-splat scene and its trackball camera controls.
    ├── impls assert isinstance(title, str)
    └── raise NotImplementedError("Dash Gaussian component assembly is declared by the skeleton but not exercised by any caller in this branch.")
```

`./data/viewer/utils/displays/gaussians/ts/backend/schemas/display_response.py`

```text
display_response.py
├── from data.viewer.utils.displays.utils.ts.backend.schemas.display_response import DisplayResponse
├── class GaussianDisplayResponse(DisplayResponse)
│   ├── # Base of the Gaussian-splat display family: a response whose served resource is a splat set.
│   ├── slot_id       # common field
│   ├── title         # common field
│   ├── display_kind  # common field
│   ├── url           # common field
│   └── meta_info     # common field
├── class ColorGSDisplayResponse(GaussianDisplayResponse)
│   ├── # Gaussian-splat display carrying per-splat RGB color.
│   ├── slot_id  # common field
│   ├── title    # common field
│   ├── display_kind = "color_gs"  # common field
│   ├── url        # common field
│   └── meta_info  # common field
└── class SegmentationGSDisplayResponse(GaussianDisplayResponse)
    ├── # Gaussian-splat display carrying per-splat class ids.
    ├── slot_id  # common field
    ├── title    # common field
    ├── display_kind = "segmentation_gs"  # common field
    ├── url        # common field
    └── meta_info  # common field
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
│   ├── # Base of the Gaussian-splat display family: a response whose served resource is a splat set.
│   ├── slot_id       # common field
│   ├── title         # common field
│   ├── display_kind  # common field
│   ├── url           # common field
│   └── meta_info     # common field
├── interface ColorGSDisplayResponse extends GaussianDisplayResponse
│   ├── # Gaussian-splat display carrying per-splat RGB color.
│   ├── slot_id  # common field
│   ├── title    # common field
│   ├── display_kind = "color_gs"  # common field
│   ├── url        # common field
│   └── meta_info  # common field
└── interface SegmentationGSDisplayResponse extends GaussianDisplayResponse
    ├── # Gaussian-splat display carrying per-splat class ids.
    ├── slot_id  # common field
    ├── title    # common field
    ├── display_kind = "segmentation_gs"  # common field
    ├── url        # common field
    └── meta_info  # common field
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
    ├── # Spatial display of camera geometry, whose centers, axes and frustums the viewer draws in the 3D slot.
    ├── slot_id  # common field
    ├── title    # common field
    ├── display_kind = "camera"  # common field
    ├── url        # common field; camera-vis JSON payload URL
    └── meta_info  # common field; empty object for camera display
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

`./data/viewer/utils/displays/cameras/ts/backend/core_camera_display.py`

```text
core_camera_display.py
└── def create_camera_display_response_core(slot_id: str, title: str, camera_vis_payload: List[Dict[str, Any]], meta_info: Optional[Dict[str, Any]] = None) -> CameraDisplayResponse
    ├── # Creates a camera display response from the already-mapped camera-vis payload, exposing it through a frontend-loadable URL.
    ├── impls serializes camera_vis_payload to a json string
    ├── impls builds the camera-vis data URL by base64-encoding that json string
    ├── impls copies caller-provided meta_info into response metadata (empty object for camera display)
    └── return
```

`./data/viewer/utils/displays/cameras/ts/frontend/types/display_response.ts`

```text
display_response.ts
├── import type { DisplayResponse } from "data/viewer/utils/displays/utils/ts/frontend/types/display_response";
└── interface CameraDisplayResponse extends DisplayResponse
    ├── # Spatial display of camera geometry, whose centers, axes and frustums the viewer draws in the 3D slot.
    ├── slot_id  # common field
    ├── title    # common field
    ├── display_kind = "camera"  # common field
    ├── url        # common field; camera-vis JSON payload URL
    └── meta_info  # common field; empty object for camera display
```

`./data/viewer/utils/displays/cameras/ts/frontend/camera_display.ts`

```text
camera_display.ts
├── import * as THREE from "three";
├── import type { LeafVNode } from "web/reconcile/reconcile";
├── import type { CameraState } from "data/viewer/utils/controls/camera/camera_state/ts/frontend/types";
├── import type { CameraDisplayResponse } from "./types/display_response";
├── import { applyCameraStateToThreeCamera, createThreeDisplayContainer, createThreePerspectiveCamera, createThreeScene, createThreeWebGLRenderer, startThreeSceneRenderLoop } from "data/viewer/utils/displays/utils/ts/frontend/three_scene_helpers";
├── const DEFAULT_FRUSTUM_OPACITY = 0.5  # number — overlay render opacity applied when the caller does not supply frustumOpacity; a dynamic render property (the per-frame hover dimming multiplies it), while glyph size + color are baked by camera_vis
├── function renderCameraDisplay({ displayResponse, initialCameraState = null, frustumOpacity, onFrameOpacityControl }: { displayResponse: CameraDisplayResponse; initialCameraState?: CameraState | null; frustumOpacity?: number; onFrameOpacityControl?: (control: CameraFrameOpacityControl) => void }): LeafVNode
│   ├── # Builds a non-interactive transparent layer from the camera-vis JSON payload (glyph sizes + colors baked by camera_vis), initialized at initialCameraState.
│   ├── throw if CameraDisplayResponse.meta_info is not an empty object
│   ├── calls createCamerasScene({ displayResponse, initialCameraState, frustumOpacity, onFrameOpacityControl })  # container, scene, camera, renderer
│   ├── calls renderCamerasScene({ scene, camera, renderer })
│   └── return LeafVNode keyed by displayResponse.url
├── function createCamerasScene({ displayResponse, initialCameraState, frustumOpacity, onFrameOpacityControl }: { displayResponse: CameraDisplayResponse; initialCameraState: CameraState | null; frustumOpacity?: number; onFrameOpacityControl?: (control: CameraFrameOpacityControl) => void }): { container: HTMLDivElement; scene: THREE.Scene; camera: THREE.PerspectiveCamera; renderer: THREE.WebGLRenderer }
│   ├── # Builds the cameras overlay scene, following the synced pose and exposing a frame-opacity control.
│   ├── calls createThreeDisplayContainer({ pointerEventsSuppressed: true })
│   ├── calls createThreeScene()
│   ├── calls createThreePerspectiveCamera({ initialCameraState })
│   ├── calls createThreeWebGLRenderer({ container })
│   ├── calls followSyncedCameraPose({ container, camera })
│   ├── impls overlay, latestPredicate start null  # the payload loads async, so the predicate is latched
│   ├── function setContributingFrames(isContributing) [local]
│   │   ├── impls latestPredicate = isContributing
│   │   └── if overlay is not null
│   │       └── calls _applyFrameOpacity({ overlay, isContributing })
│   ├── if onFrameOpacityControl is defined
│   │   └── impls onFrameOpacityControl({ container, setContributingFrames })
│   ├── impls loadCamerasPayload({ displayResponse }).then(payload => { overlay = createThreeCameras({ payload, frustumOpacity }); scene.add(overlay); if (latestPredicate !== null) _applyFrameOpacity({ overlay, isContributing: latestPredicate }) })
│   ├── impls that promise's .catch(error => container.replaceChildren(_renderCamerasStatus(`Failed to load camera visualization: ${message}`)))
│   └── return  # { container, scene, camera, renderer }
├── async function loadCamerasPayload({ displayResponse }: { displayResponse: CameraDisplayResponse }): Promise<CamerasPayload>
│   ├── # Async-fetches the camera-vis JSON payload from displayResponse.url and hands the decoded body to the payload validator.
│   ├── if displayResponse.url === null
│   │   └── throw new Error("camera display response url is null")
│   ├── impls response = await fetch(displayResponse.url)
│   ├── if !response.ok
│   │   └── throw new Error(`unable to load camera visualization: HTTP ${response.status}`)
│   └── return validateCameraVisualizationPayloads({ value: await response.json() })
├── function createThreeCameras({ payload, frustumOpacity }: { payload: CamerasPayload; frustumOpacity?: number }): THREE.Object3D
│   ├── # Sync-builds the transparent Three.js centers + line segments from a pre-validated camera-vis payload, reading every baked glyph size + color from the payload.
│   ├── impls effectiveFrustumOpacity = frustumOpacity ?? DEFAULT_FRUSTUM_OPACITY
│   ├── impls overlay = a THREE.Group carrying cameraCount, lineCount and renderOrder 999  # impls-node-one-step:skip
│   ├── for each cameraVisualization in payload
│   │   ├── impls cameraGroup = a THREE.Group tagged with its cameraIndex
│   │   ├── calls createThreeCameraCenter({ cameraVisualization })
│   │   ├── for each line of cameraVisualization.axes
│   │   │   └── calls createThreeCameraOverlayLine({ line, frustumOpacity: effectiveFrustumOpacity })
│   │   ├── for each line of cameraVisualization.frustum_lines
│   │   │   └── calls createThreeCameraOverlayLine({ line, frustumOpacity: effectiveFrustumOpacity })
│   │   └── impls overlay.add(cameraGroup), lineCount rising with each added line
│   ├── if payload is non-empty
│   │   ├── calls cameraVisualizationLineLength({ line: payload[0].axes[0] })           # firstAxisLength
│   │   └── calls cameraVisualizationLineLength({ line: payload[0].frustum_lines[0] })  # firstFrustumLength
│   └── return overlay
├── function renderCamerasScene({ scene, camera, renderer }: { scene: THREE.Scene; camera: THREE.PerspectiveCamera; renderer: THREE.WebGLRenderer }): void
│   ├── # Drives the render loop; the cameras-overlay has no trackball controls — its camera is externally synced through the camera-sync registry observing the display element's data-camera-state attribute.
│   ├── impls exposes the display element under displayResponse.slot_id so the caller can register it as a camera-sync target
│   ├── calls startThreeSceneRenderLoop({ scene, camera, renderer, controls: null })
│   └── return
├── function validateCameraVisualizationPayloads({ value }: { value: unknown }): CamerasPayload
│   ├── # Validates the decoded camera-vis JSON body into the typed per-camera payload array.
│   ├── if !Array.isArray(value)
│   │   └── throw new Error("camera visualization payload must be an array")
│   ├── for each cameraVisualization, cameraIndex of value
│   │   └── calls validateCameraVisualizationPayload({ value: cameraVisualization, cameraIndex })
│   ├── calls assertCameraVisualizationPayloadShape({ cameraVisualizations })
│   └── return cameraVisualizations
├── function validateCameraVisualizationPayload({ value, cameraIndex }: { value: unknown; cameraIndex: number }): CameraVisualizationPayload
│   ├── # Validates one camera entry into its center / center_color / center_size / axes / frustum_lines fields.
│   ├── if !isRecord(value)
│   │   └── throw new Error(`camera visualization entry must be an object: ${cameraIndex}`)
│   ├── calls validateCameraVisualizationVector({ value: value.center, label: `camera ${cameraIndex} center` })               → center
│   ├── calls validateCameraVisualizationVector({ value: value.center_color, label: `camera ${cameraIndex} center_color` })   → center_color
│   ├── calls validateCameraVisualizationScalar({ value: value.center_size, label: `camera ${cameraIndex} center_size` })     → center_size
│   ├── calls validateCameraVisualizationLines({ value: value.axes, label: `camera ${cameraIndex} axes` })                    → axes
│   ├── calls validateCameraVisualizationLines({ value: value.frustum_lines, label: `camera ${cameraIndex} frustum_lines` })  → frustum_lines
│   └── return { center, center_color, center_size, axes, frustum_lines }
├── function validateCameraVisualizationLines({ value, label }: { value: unknown; label: string }): CameraVisualizationLinePayload[]
│   ├── # Validates one axes or frustum_lines array line by line.
│   ├── if !Array.isArray(value)
│   │   └── throw new Error(`${label} must be an array`)
│   ├── for each lineValue, lineIndex of value
│   │   └── calls validateCameraVisualizationLine({ value: lineValue, label: `${label} line ${lineIndex}` })
│   └── return the validated line array
├── function validateCameraVisualizationLine({ value, label }: { value: unknown; label: string }): CameraVisualizationLinePayload
│   ├── # Validates one axes or frustum line into its start / end / color fields.
│   ├── if !isRecord(value)
│   │   └── throw new Error(`${label} must be an object`)
│   ├── calls validateCameraVisualizationVector({ value: value.start, label: `${label} start` })  → start
│   ├── calls validateCameraVisualizationVector({ value: value.end, label: `${label} end` })      → end
│   ├── calls validateCameraVisualizationVector({ value: value.color, label: `${label} color` })  → color
│   └── return { start, end, color }
├── function validateCameraVisualizationVector({ value, label }: { value: unknown; label: string }): CameraVisualizationVectorPayload
│   ├── # Validates one payload field as a finite 3-vector.
│   ├── if !Array.isArray(value) || value.length !== 3 || value.some(entry => typeof entry !== "number" || !Number.isFinite(entry))
│   │   └── throw new Error(`${label} must be a finite 3-vector`)
│   └── return [value[0], value[1], value[2]]
├── function validateCameraVisualizationScalar({ value, label }: { value: unknown; label: string }): number
│   ├── # Validates one payload field as a finite number.
│   ├── if typeof value !== "number" || !Number.isFinite(value)
│   │   └── throw new Error(`${label} must be a finite number`)
│   └── return value
├── function assertCameraVisualizationPayloadShape({ cameraVisualizations }: { cameraVisualizations: CamerasPayload }): void
│   ├── # Holds every validated camera to exactly three axes and eight frustum lines.
│   ├── for each cameraVisualization, index of cameraVisualizations
│   │   ├── if cameraVisualization.axes.length !== 3
│   │   │   └── throw new Error(`camera ${index} must contain three axes`)
│   │   └── if cameraVisualization.frustum_lines.length !== 8
│   │       └── throw new Error(`camera ${index} must contain eight frustum lines`)
│   └── return
├── function isRecord(value: unknown): value is Record<string, unknown>
│   ├── # Narrows an unknown decoded JSON value to a keyed object.
│   └── return typeof value === "object" && value !== null
├── function followSyncedCameraPose({ container, camera }: { container: HTMLDivElement; camera: THREE.PerspectiveCamera }): void
│   ├── # Mirrors the container's data-camera-state onto the camera, now and on every later mutation.
│   ├── function applyDatasetCameraState() [local]
│   │   ├── # Applies the container's data-camera-state, ignoring an unset or malformed value.
│   │   ├── impls raw = container.dataset.cameraState
│   │   ├── if raw is undefined
│   │   │   └── return
│   │   ├── try
│   │   │   └── calls applyCameraStateToThreeCamera({ camera, cameraState: JSON.parse(raw) as CameraState })
│   │   └── catch
│   │       └── impls unparseable dataset values are ignored
│   ├── calls applyDatasetCameraState()  # the initial pose
│   └── impls a MutationObserver on the data-camera-state attribute calls it again on every change
├── function _applyFrameOpacity({ overlay, isContributing }: { overlay: THREE.Object3D; isContributing: (cameraIndex: number) => boolean }): void
│   ├── # Dims every non-contributing camera group to a fixed ratio of its own base opacity.
│   └── for each child of overlay
│       ├── if its userData.cameraIndex is not a number
│       │   └── impls continue
│       ├── impls contributing = isContributing(cameraIndex)
│       └── impls child.traverse(object => { skip absent or array materials; capture userData.baseOpacity once; transparent = true; opacity = contributing ? baseOpacity : baseOpacity * NON_CONTRIBUTING_FRAME_OPACITY_RATIO; needsUpdate = true })
├── function createThreeCameraCenter({ cameraVisualization }: { cameraVisualization: CameraVisualizationPayload }): THREE.Points
│   ├── # Builds the depth-test-free points object marking one camera's center.
│   ├── impls geometry carries cameraVisualization.center as a 3-component position attribute
│   ├── impls material = a THREE.PointsMaterial in center_color at center_size, depth-test and depth-write off, sizeAttenuation on  # impls-node-one-step:skip
│   └── return the THREE.Points at renderOrder 999
├── function createThreeCameraOverlayLine({ line, frustumOpacity }: { line: CameraVisualizationLinePayload; frustumOpacity: number }): THREE.Line
│   ├── # Builds one depth-test-free overlay line in the color the backend baked into it.
│   ├── impls geometry carries line.start followed by line.end as a 3-component position attribute
│   ├── impls material = a THREE.LineBasicMaterial in line.color at frustumOpacity, transparent, depth-test and depth-write off  # impls-node-one-step:skip
│   └── return the THREE.Line at renderOrder 999
├── function cameraVisualizationLineLength({ line }: { line: CameraVisualizationLinePayload }): number
│   ├── # Returns the Euclidean length of one visualization line.
│   └── return the square root of the summed squared start-to-end deltas
└── function _renderCamerasStatus(message: string): HTMLElement
    ├── # Builds the centered, italic status surface shown when the payload fails to load.
    ├── impls status = a "camera-display-scene__status" div, flex-centered at 100% by 100%, padded 1rem,  # 888 italic
    ├── impls status.textContent = message
    └── return status
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
├── from pydantic import BaseModel
└── class CameraState(BaseModel)
    ├── # One camera's viewer-side state: its intrinsics, extrinsics and convention, plus the name and id identifying it.
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
    ├── # One camera's viewer-side state: its intrinsics, extrinsics and convention, plus the name and id identifying it.
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
│   ├── impls Dash renderer-specific trackball camera controls with left-button rotation, right-button panning, mouse-wheel zoom, and suppressed canvas context menu  # impls-node-one-step:skip
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
│   ├── # The trackball camera-control contract: serializing the full camera state, applying one, and subscribing to its changes.
│   ├── getCameraState
│   │   └── # serializes the entire camera state (every CameraState field — both intrinsics and extrinsics) into a CameraState
│   ├── applyCameraState
│   │   └── # applies the entire CameraState (every field — both intrinsics and extrinsics) to the underlying camera and controls
│   └── subscribeCameraStateChange
├── function createTrackballCameraControls
│   ├── # Builds, validates, and returns the trackball controls, seeding them from initialCameraState and observing the container's data-camera-state attribute for external sync.
│   ├── if "camera" in args
│   │   └── return createThreeTrackballCameraControls(args)
│   ├── calls createRendererTrackballCameraControls({ targetElement, initialCameraState })
│   ├── calls assertTrackballCameraControls(controls)
│   └── return controls
├── function createRendererTrackballCameraControls
│   ├── # Constructs the renderer-specific trackball controls wiring left-drag rotate, right-drag pan, wheel zoom, and context-menu suppression.
│   ├── impls the control latches currentCameraState, internallyWrittenCameraStateToken, a listeners array
│   ├── function setInternallyWrittenCameraStateToken(token) [local]
│   │   └── impls internallyWrittenCameraStateToken = token
│   ├── function applyCameraState(cameraState) [local]
│   │   ├── impls currentCameraState = cameraState
│   │   ├── calls writeInternalCameraStateToTargetElement({ targetElement, cameraState, setInternallyWrittenCameraStateToken })
│   │   └── calls postCameraStateToEmbeddedRenderer({ targetElement, cameraState })
│   ├── function emitCameraStateChange(cameraState) [local]
│   │   ├── impls currentCameraState = cameraState
│   │   ├── calls writeInternalCameraStateToTargetElement({ targetElement, cameraState, setInternallyWrittenCameraStateToken })
│   │   ├── impls every registered listener receives that state
│   │   └── impls targetElement dispatches a bubbling "camera-pose-change" CustomEvent carrying it
│   ├── impls the mutationObserver callback = () => applyExternalCameraState(readCameraStateFromTargetElement(targetElement)), returning early when readCameraStateTokenFromTargetElement(targetElement) echoes this control's own write, with internallyWrittenCameraStateToken cleared either way
│   ├── impls mutationObserver observes targetElement for `data-camera-state` attribute changes
│   ├── impls window.addEventListener("message", event => emitCameraStateChange(message.cameraState)) — skipped unless isEmbeddedRendererMessageSource({ targetElement, source: event.source }), event.origin is this window's, isTrackballCameraStateChangeMessage(event.data)
│   ├── if targetElement is an HTMLIFrameElement
│   │   └── impls its "load" event re-posts currentCameraState to the embedded renderer
│   ├── impls targetElement.dataset takes cameraControlMode "trackball", trackballMouseMapping, contextMenuBehavior "suppressed-for-trackball-pan"
│   ├── if currentCameraState is not null
│   │   └── calls applyCameraState(currentCameraState)
│   ├── function applyExternalCameraState(cameraState) [local]
│   │   ├── impls currentCameraState = cameraState
│   │   └── calls postCameraStateToEmbeddedRenderer({ targetElement, cameraState })
│   ├── function getCameraState() [local]
│   │   ├── # The returned object's getCameraState: the last state this control saw.
│   │   └── return currentCameraState
│   ├── function subscribeCameraStateChange(listener) [local]
│   │   ├── # The returned object's subscribeCameraStateChange: registers a listener, handing back its unsubscribe.
│   │   ├── if listener is not a function
│   │   │   └── throw new Error("camera state listener must be a function")
│   │   ├── impls listeners.push(listener)
│   │   └── return  # an unsubscribe splicing it out of listeners
│   └── return  # { targetElement, getCameraState, applyCameraState, subscribeCameraStateChange }
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
├── function assertNoCameraPoseClamps
│   ├── # Asserts the controls impose no camera-pose restriction on polar angle, azimuth angle, target lock, distance, pan, translation, or rotation.
│   ├── if controls restrict polar angle, azimuth angle, target lock, distance bounds, pan, translation, or rotation
│   │   └── throw restricted camera pose controls
│   └── return
├── function createThreeTrackballCameraControls(args: { camera: THREE.PerspectiveCamera; renderer: THREE.WebGLRenderer; container: HTMLElement; initialCameraState?: CameraState | null }): ThreeTrackballCameraControls
│   ├── # Wraps THREE's trackball controls, publishing a CameraState to its listeners on every change.
│   ├── impls threeControls = a ThreeTrackballControlsImpl over the renderer's canvas
│   ├── impls rotateSpeed 3, zoomSpeed 1.5, panSpeed 0.8, staticMoving on
│   ├── impls the canvas contextmenu event is prevented
│   ├── impls listeners = a Set of CameraStateListener
│   ├── impls threeControls.addEventListener("change", () => every registered listener receives buildThreeTrackballCameraState({ camera, controls: threeControls }))
│   ├── function getCameraState() [local]
│   │   ├── # result.getCameraState: the pose read fresh from the camera with its controls.
│   │   └── calls buildThreeTrackballCameraState({ camera, controls: threeControls })
│   ├── function applyCameraState(cameraState) [local]
│   │   ├── # result.applyCameraState: writes one pose onto the camera with its controls.
│   │   └── calls applyThreeTrackballCameraState({ camera, controls: threeControls, cameraState })
│   ├── function subscribeCameraStateChange(listener) [local]
│   │   ├── # result.subscribeCameraStateChange: registers a listener, handing back its unsubscribe.
│   │   ├── if listener is not a function
│   │   │   └── throw new Error("camera state listener must be a function")
│   │   ├── impls listeners.add(listener)
│   │   └── return  # an unsubscribe deleting it from listeners
│   ├── impls result = Object.assign(threeControls, { getCameraState, applyCameraState, subscribeCameraStateChange })
│   ├── if initialCameraState is not null
│   │   └── calls result.applyCameraState(initialCameraState)
│   ├── impls the observer callback = () => result.applyCameraState(JSON.parse(container.dataset.cameraState) as CameraState), ignoring an undefined or unparseable value
│   ├── impls observer observes container for `data-camera-state` attribute changes
│   └── return result
├── function buildThreeTrackballCameraState({ camera, controls }: { camera: THREE.PerspectiveCamera; controls: ThreeTrackballControlsImpl }): CameraState
│   ├── # Reads the camera and its controls into a "three_trackball" CameraState.
│   ├── calls vectorToRecord(camera.position)
│   ├── calls quaternionToRecord(camera.quaternion)
│   ├── calls vectorToRecord(controls.target)
│   ├── calls vectorToRecord(camera.up)
│   └── return  # intrinsics aspect/far/fov/near at projection "perspective-three", those extrinsics, convention "three_trackball", name and id null
├── function applyThreeTrackballCameraState({ camera, controls, cameraState }: { camera: THREE.PerspectiveCamera; controls: ThreeTrackballControlsImpl; cameraState: CameraState | null }): void
│   ├── # Overlays a "three_trackball" CameraState onto the camera and its controls, ignoring any other convention.
│   ├── if cameraState is null or its convention is not "three_trackball"
│   │   └── return
│   ├── calls isVectorRecord(position)
│   ├── calls isQuaternionRecord(quaternion)
│   ├── calls isVectorRecord(target)
│   ├── calls isVectorRecord(up)
│   ├── if any of those fails, or aspect, far, fov or near is not a number
│   │   └── return
│   ├── impls the camera's position, quaternion, up and intrinsics, and the controls' target, take those values  # impls-node-one-step:skip
│   ├── impls camera.updateProjectionMatrix()
│   └── impls controls.update()
├── function readCameraStateFromTargetElement(targetElement: HTMLElement): CameraState | null
│   ├── # Parses the target element's data-camera-state, rejecting a payload that is not a CameraState.
│   ├── if data-camera-state is unset
│   │   └── return null
│   ├── calls isCameraState(the parsed value)
│   ├── if it is not a CameraState
│   │   └── impls throw new Error("target camera state does not match CameraState")
│   └── return that parsed state
├── function writeInternalCameraStateToTargetElement(args: { targetElement: HTMLElement; cameraState: CameraState | null; setInternallyWrittenCameraStateToken: (token: string | null) => void }): void
│   ├── # Writes a state to the element and records its token, so the write is not mistaken for an external one.
│   ├── calls serializeCameraState(cameraState)
│   ├── calls writeCameraStateToTargetElement({ targetElement, cameraState, serializedCameraState })
│   └── if that write happened
│       └── impls setInternallyWrittenCameraStateToken(serializedCameraState)
├── function writeCameraStateToTargetElement(args: { targetElement: HTMLElement; cameraState: CameraState | null; serializedCameraState: string | null }): boolean
│   ├── # Writes the serialized state onto the element, reporting whether anything changed.
│   ├── calls readCameraStateTokenFromTargetElement(targetElement)
│   ├── if that token already equals serializedCameraState
│   │   └── return false
│   ├── if cameraState is null
│   │   ├── impls delete targetElement.dataset.cameraState
│   │   └── return true
│   ├── if serializedCameraState is null
│   │   └── impls throw new Error("serialized camera state is unexpectedly null")
│   └── return true, having written serializedCameraState onto the element
├── function postCameraStateToEmbeddedRenderer(args: { targetElement: HTMLElement; cameraState: CameraState | null }): void
│   ├── # Forwards a state to an embedded renderer iframe, same-origin only.
│   ├── if targetElement is not an HTMLIFrameElement, or has no contentWindow
│   │   └── return
│   └── impls targetWindow.postMessage({ cameraState, type: "trackball-camera-state" }, window.location.origin)
├── function isEmbeddedRendererMessageSource(args: { targetElement: HTMLElement; source: MessageEventSource | null }): boolean
│   ├── # A message is the embedded renderer's when its source is that iframe's own contentWindow.
│   └── return that predicate
├── function isTrackballCameraStateChangeMessage(value: unknown): value is { type: "trackball-camera-state-change"; cameraState: CameraState }
│   ├── # Narrows a posted message to the trackball camera-state-change shape.
│   ├── calls isRecord(value)
│   ├── calls isCameraState(value.cameraState)
│   └── return that, and whether type is "trackball-camera-state-change"
├── function isCameraState(value: unknown): value is CameraState
│   ├── # A CameraState carries record intrinsics and extrinsics, a string convention, and nullable name and id.
│   ├── calls isRecord(value)
│   └── return that predicate
├── function serializeCameraState(cameraState: CameraState | null): string | null
│   ├── # Serializes a state to JSON, passing null through.
│   ├── if cameraState is null
│   │   └── return null
│   └── return JSON.stringify(cameraState)
├── function readCameraStateTokenFromTargetElement(targetElement: HTMLElement): string | null
│   ├── # Returns the element's raw data-camera-state token, or null when unset.
│   └── return targetElement.dataset.cameraState ?? null
├── function vectorToRecord(vector: THREE.Vector3): Record<string, number>
│   ├── # Flattens a Vector3 into an x/y/z record.
│   └── return { x: vector.x, y: vector.y, z: vector.z }
├── function quaternionToRecord(quaternion: THREE.Quaternion): Record<string, number>
│   ├── # Flattens a Quaternion into an x/y/z/w record.
│   └── return { x: quaternion.x, y: quaternion.y, z: quaternion.z, w: quaternion.w }
├── function isQuaternionRecord(value: unknown): value is { x: number; y: number; z: number; w: number }
│   ├── # A quaternion record is a vector record that also carries a numeric w.
│   ├── calls isVectorRecord(value)
│   └── return that, and whether w is a number
├── function isVectorRecord(value: unknown): value is { x: number; y: number; z: number }
│   ├── # A vector record is a non-null object whose x, y and z are all numbers.
│   └── return that predicate
└── function isRecord(value: unknown): value is Record<string, unknown>
    ├── # A record is any non-null object.
    └── return whether value is non-null and of type object
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
│   ├── impls assert source_camera is None or isinstance(source_camera, dict)
│   ├── impls assert camera_sync_state is None or isinstance(camera_sync_state, dict)
│   ├── impls assert isinstance(source_id, (str, dict))
│   ├── if camera_sync_state is None
│   │   └── impls updated_camera_sync_state = {"camera_state": None, "source_id": None, "target_ids": []}
│   ├── else
│   │   └── impls updated_camera_sync_state = dict(camera_sync_state)
│   ├── impls updated_camera_sync_state["camera_state"] = source_camera  # committed even when None
│   ├── impls updated_camera_sync_state["source_id"] = source_id
│   └── return updated_camera_sync_state  # the updated camera-sync store data
└── def apply_camera_state_to_target
    ├── # Applies one source's current camera state to a single registered Dash spatial-display target.
    ├── impls applies the source's CameraSyncState.camera_state to a Dash spatial-display target registered under that source
    └── return
```

`./data/viewer/utils/controls/camera/camera_sync/ts/frontend/types.ts`

```text
types.ts
└── interface CameraSyncState
    ├── # One source's camera-sync entry: the source id, the targets registered under it, and that source's current camera state.
    ├── source_id     # the source this entry belongs to; one CameraSyncState exists per source
    ├── target_ids    # targets registered under this source
    └── camera_state  # this source's current camera state
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
│   │   ├── impls this._state_by_source_id[source_id] = { target_ids: empty, camera_state: the caller-provided CameraState }
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
│   │   ├── calls this._apply_camera_state_to_element(target_element, this._state_by_source_id[source_id].camera_state)
│   │   └── return
│   ├── unregisterCameraSyncTarget
│   │   ├── # Additional API: unregisters one display panel from a source's target set.
│   │   ├── impls idempotently deletes this._targets_by_source_id[source_id].delete(target_id)
│   │   ├── impls updates this._state_by_source_id[source_id].target_ids from this._targets_by_source_id[source_id].keys()
│   │   └── return
│   ├── applyCameraSyncStateToTargets
│   │   ├── # Additional API: applies a caller-owned CameraState to every target registered under one source.
│   │   ├── impls this._state_by_source_id[source_id] = { target_ids: the current target_ids, camera_state: the caller-provided CameraState }
│   │   ├── for each (target_id, target_element) in this._targets_by_source_id[source_id]
│   │   │   └── calls this._apply_camera_state_to_element(target_element, camera_state)
│   │   ├── calls this._emit_camera_sync_state(this._state_by_source_id[source_id])
│   │   └── return
│   ├── applySourceCameraStateToTargets
│   │   ├── # Additional API: ingests camera movement from a source display and propagates it to that source's other registered targets.
│   │   ├── if source_id not in this._targets_by_source_id
│   │   │   └── throw
│   │   ├── impls this._state_by_source_id[source_id] = { target_ids: the current target_ids, camera_state: the source display CameraState }
│   │   ├── for each (target_id, target_element) in this._targets_by_source_id[source_id]
│   │   │   ├── if target_id == source_id
│   │   │   │   └── continue
│   │   │   └── calls this._apply_camera_state_to_element(target_element, camera_state)
│   │   ├── calls this._emit_camera_sync_state(this._state_by_source_id[source_id])
│   │   └── return
│   ├── _apply_camera_state_to_element
│   │   ├── # Writes a CameraState onto an element's `data-camera-state` attribute; mesh / point-cloud display containers observe this attribute and re-apply to their trackball controls.
│   │   └── impls sets target_element.dataset.cameraState to the serialized CameraState (or deletes the attribute when CameraState is null)
│   └── _emit_camera_sync_state
│       ├── # Notifies every subscriber with the just-updated source's CameraSyncState.
│       └── for each listener in this._listeners
│           └── impls listener(camera_sync_state)
└── const cameraSyncRegistry = new CameraSyncRegistry()  # the single document-global registry instance shared by every spatial display in the document; consumers import this instance and call its methods
```

`./data/viewer/utils/controls/selectors/ts/backend/schemas/selector_response.py`

```text
selector_response.py
├── from typing import List
├── from pydantic import BaseModel
├── def build_selector_response
│   ├── # Build a SelectorResponse from an app's nested (value, label, children) option tuple — the app owns the tree shape, the lib owns the schema.
│   ├── calls _to_selection_node(option_tree)
│   └── return  # SelectorResponse(root=converted imaginary root)
├── def _to_selection_node
│   ├── # Recursion helper: convert one (value, label, children) tuple into a SelectionNode, recursing into each child tuple.
│   ├── for each child tuple
│   │   └── calls _to_selection_node
│   ├── calls SelectionNode
│   └── return  # a SelectionNode holding its converted children
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
    ├── # Complete a selector level change into a full root-leaf path, resetting every finer level to its first option.
    ├── impls start the path with the prefix up to the chosen level plus the chosen value
    ├── for each deeper level until the descended node has no children
    │   ├── impls append the descended node's first child's value
    │   └── impls descend into that first child
    └── return  # the completed root-leaf path
```

`./data/viewer/utils/controls/selectors/ts/frontend/selector_cascade.ts`

```text
selector_cascade.ts
├── import type { ElementVNode, LeafVNode } from "web/reconcile/reconcile";
├── import type { SelectorResponse, SelectionNode } from "data/viewer/utils/controls/selectors/ts/frontend/types/selector_response";
├── import { completeRootLeafPath } from "data/viewer/utils/controls/selectors/ts/frontend/selection_path";
├── function renderSelectorCascade({ axisKey, response, path, onPathChange }: { axisKey: string; response: SelectorResponse; path: string[]; onPathChange: (next: string[]) => void }): ElementVNode
│   ├── # Render one selector axis as a cascade of native <select> dropdowns, one per level descended from the response's imaginary root down to a leaf.
│   ├── calls _renderSelectorLevel({ node: response.root, level: 0, axisKey, path, onPathChange })  # collect the per-level <select> leaves from the imaginary root down
│   └── return  # a container ElementVNode wrapping the collected <select> leaves
└── function _renderSelectorLevel({ node, level, axisKey, path, onPathChange }: { node: SelectionNode; level: number; axisKey: string; path: string[]; onPathChange: (next: string[]) => void }): LeafVNode[]
    ├── # Recursion helper: collect the <select> leaves from this level down; the base case (a node with no children) contributes none.
    ├── if node has no children
    │   └── return  # [] — base case: a leaf level adds no dropdown
    ├── impls the <select> is a reconciler leaf keyed `${axisKey}-select-${level}-${path[level-1] ?? "root"}` (its option-set identity) so a coarser-level change re-mounts it with this parent's children
    ├── impls build a native <select> over node's children
    ├── function _onLevelChange [local]
    │   ├── # The <select> change handler: report the completed root-leaf path to onPathChange.
    │   ├── calls completeRootLeafPath
    │   └── calls onPathChange
    ├── calls _onLevelChange  # bound as the <select>'s change listener
    ├── calls _renderSelectorLevel({ node: selectedChild, level: level + 1, axisKey, path, onPathChange })  # recurse into the path-selected child to collect the deeper levels' leaves
    └── return  # [this level's <select> leaf, ...the deeper levels' leaves]
```

`./data/viewer/utils/controls/selectors/dash/selector_cascade.py`

```text
selector_cascade.py
├── from typing import List
├── from data.viewer.utils.controls.selectors.ts.backend.schemas.selector_response import SelectorResponse, SelectionNode
├── def render_selector_cascade(response: SelectorResponse, path: List[str])
│   ├── # Render one selector axis as a Dash cascade of dropdowns from a SelectorResponse and the current path: one dropdown per level, descending the imaginary root along the path to a leaf, re-rendered per parent change.
│   ├── calls _render_selector_level
│   └── return  # the dropdown-stack Dash component
├── def _render_selector_level(node: SelectionNode, level: int, path: List[str])
│   ├── # Recursion helper: a Dash dropdown over this node's children, then recurse into the child the path selects, stopping at a leaf.
│   ├── if this node has children
│   │   └── calls _render_selector_level
│   └── return
└── def complete_root_leaf_path(node: SelectionNode, path: List[str])
    ├── # Complete a Dash level change into a full root-leaf path: the chosen value, then each deeper level's first child descended to a leaf.
    ├── for each deeper level until the descended node has no children
    │   ├── impls append the descended node's first child's value
    │   └── impls descend into that first child
    └── return  # the completed root-leaf path
```

`./data/viewer/utils/displays/aabbs/threed/ts/backend/schemas/display_response.py`

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
    ├── # Spatial overlay response: inline axis-aligned 3D boxes (each a 6-float box) with optional per-box scores, composed as an aux layer over a point cloud.
    ├── slot_id  # common field
    ├── title    # common field
    ├── display_kind = "aabb_3d"  # common field
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
│   ├── calls _boxesBoundingRadius({ boxes })  # boundingRadius
│   ├── impls group = new THREE.Group()
│   ├── for each box in displayResponse.aabbs
│   │   ├── impls boxGroup = new THREE.Group()
│   │   ├── calls _createBoxLines({ box })
│   │   ├── if displayResponse.scores is not null
│   │   │   ├── calls _createScoreLabelSprite({ score: scores[boxIndex] })
│   │   │   └── impls the sprite sits at the box's top-face center, scaled by boundingRadius * AABB_3D_LABEL_HEIGHT_RATIO, renderOrder 1001
│   │   └── impls group.add(boxGroup)
│   └── return group
├── function renderAabb3dScene({ scene, camera, renderer, controls }: { scene: THREE.Scene; camera: THREE.PerspectiveCamera; renderer: THREE.WebGLRenderer; controls: ReturnType<typeof createTrackballCameraControls> }): void
│   ├── # Drives the 3D-box display render loop with the supplied trackball controls.
│   ├── calls startThreeSceneRenderLoop({ scene, camera, renderer, controls })
│   └── return
├── function _boxesBoundingRadius({ boxes }: { boxes: number[][] }): number
│   ├── # Returns the bounding-sphere radius of all boxes, falling back to 1.
│   ├── if boxes is empty
│   │   └── return 1
│   ├── impls boundingBox = a THREE.Box3 expanded by every box's min and max corner  # impls-node-one-step:skip
│   ├── impls sphere = boundingBox.getBoundingSphere(...)
│   └── return sphere.radius when positive, else 1
├── function _createBoxLines({ box }: { box: number[] }): LineSegments2
│   ├── # Builds the twelve-edge wireframe of one axis-aligned box.
│   ├── impls corners = the box's eight (min/max) corner triples
│   ├── impls edges = the twelve corner-index pairs spanning the box
│   ├── impls positions = the edge endpoints flattened onto a LineSegmentsGeometry
│   ├── impls material = a LineMaterial in AABB_3D_BOX_COLOR at AABB_3D_BOX_LINEWIDTH, transparent, depth-test off, resolution from the window  # WebGL ignores LineBasicMaterial.linewidth
│   └── return the LineSegments2 built from them, at renderOrder 1000
├── function _createScoreLabelSprite({ score }: { score: number }): THREE.Sprite
│   ├── # Draws one score onto a 256x64 canvas and wraps it as a depth-test-free sprite.
│   ├── impls canvas = a 256 by 64 canvas
│   ├── if the canvas 2d context is null
│   │   └── impls throw new Error("aabb 3d score label canvas 2d context is unavailable")
│   ├── impls score.toFixed(2) is written in bold 36px monospace  # ffffff over an rgba(77,166,255,0.85) fill
│   ├── impls spriteMaterial = new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(canvas), depthTest: false, transparent: true })
│   └── return new THREE.Sprite(spriteMaterial)
└── impls registerSpatialLayerRenderer({ displayKind: "aabb_3d", layerRenderer: createAabb3dObject })  # module-load self-registration of the spatial aabb-3d layer renderer
```

`./data/viewer/utils/displays/aabbs/twod/ts/backend/schemas/display_response.py`

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
    ├── # Raster overlay response: inline axis-aligned 2D boxes (each a 4-float box) with optional per-box scores, composed as an aux layer over an image.
    ├── slot_id  # common field
    ├── title    # common field
    ├── display_kind = "aabb_2d"  # common field
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
│   ├── calls _buildBoxesOverlay({ displayResponse })  # the leaf's render()
│   └── return LeafVNode keyed by displayResponse.url
├── function _buildBoxesOverlay({ displayResponse }: { displayResponse: Aabb2dDisplayResponse }): HTMLElement
│   ├── # Builds the full-bleed SVG box overlay and its optional per-box score labels.
│   ├── impls overlay = an absolutely-positioned, inset-0, pointer-events-none div
│   ├── impls svg = an SVG element with preserveAspectRatio="none", sized 100% by 100%
│   ├── impls overlay.append(svg)
│   ├── for each box in displayResponse.aabbs
│   │   ├── impls rect = an SVG rect at (x1, y1) sized (x2 - x1) by (y2 - y1), stroked AABB_2D_BOX_STROKE with no fill
│   │   └── impls the box's score, when scores is not null, becomes a text label beside the rect
│   └── return overlay
└── impls registerRasterLayerRenderer({ displayKind: "aabb_2d", layerRenderer: renderAabb2dDisplay })  # module-load self-registration of the raster aabb-2d layer renderer
```
