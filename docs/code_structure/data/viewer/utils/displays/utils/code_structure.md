# Data Viewer Display Utils Code Structure

## 1. Code structure trees

`data/viewer/utils/displays/utils/class_colors.py`

```text
class_colors.py
├── from typing import Dict, Tuple
├── import torch
├── def map_class_ids_to_rgb(class_ids: torch.Tensor) -> Dict[int, Tuple[int, int, int]]
│   ├── # Maps each distinct class id to a deterministic RGB color from a fixed class-color palette.
│   ├── impls assert isinstance(class_ids, torch.Tensor)
│   ├── impls assert class_ids.numel() > 0
│   ├── impls flattened_class_ids = class_ids.detach().cpu().reshape(-1).to(torch.int64)
│   ├── impls unique_class_ids = torch.unique(flattened_class_ids, sorted=True)
│   ├── calls get_class_color(class_id=each unique class id cast to int)
│   ├── impls class_id_to_rgb = each such int keyed to its own get_class_color result
│   └── return class_id_to_rgb
└── def get_class_color(class_id: int) -> Tuple[int, int, int]
    ├── # Maps one class identifier onto a stable palette color, wrapping the palette for ids past its end.
    ├── impls assert isinstance(class_id, int)
    ├── impls assert class_id >= 0
    ├── impls palette = [(37, 99, 235), (220, 38, 38), (22, 163, 74), (202, 138, 4), (147, 51, 234), (8, 145, 178), (234, 88, 12), (79, 70, 229)]
    └── return palette[class_id % len(palette)]
```

`data/viewer/utils/displays/utils/heatmap_colors.py`

```text
heatmap_colors.py
├── import torch
└── def map_scalars_to_rgb(scalars: torch.Tensor) -> torch.Tensor
    ├── # Maps non-negative scalars to RGB via a fixed continuous heatmap palette.
    ├── assert scalars is non-negative
    └── return torch.Tensor of shape (*scalars.shape, 3)
```

### Backend schemas

`data/viewer/utils/displays/utils/ts/backend/schemas/display_response.py`

```text
display_response.py
├── from pydantic import BaseModel
└── class DisplayResponse(BaseModel)
    ├── slot_id       # common field
    ├── title         # common field
    ├── display_kind  # common field
    ├── url           # common field
    └── meta_info     # common field
```

### Frontend

`data/viewer/utils/displays/utils/ts/frontend/types/display_response.ts`

```text
display_response.ts
└── interface DisplayResponse
    ├── slot_id       # common field
    ├── title         # common field
    ├── display_kind  # common field
    ├── url           # common field
    └── meta_info     # common field
```

### Backend schemas

`data/viewer/utils/displays/utils/ts/backend/schemas/layered_display_response.py`

```text
layered_display_response.py
├── from typing import List, Literal
├── from data.viewer.utils.displays.utils.ts.backend.schemas.display_response import DisplayResponse
├── RASTER_DISPLAY_KINDS   # frozenset[str]: color_image, depth_image, edge_image, normal_image, segmentation_image, instance_surrogate_image, video, aabb_2d — the single source of the raster/spatial taxonomy
├── SPATIAL_DISPLAY_KINDS  # frozenset[str]: color_pc, segmentation_pc, color_gs, segmentation_gs, scene_graph, camera, aabb_3d
└── class LayeredDisplayResponse(DisplayResponse)
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

### Frontend

`data/viewer/utils/displays/utils/ts/frontend/types/layered_display_response.ts`

```text
layered_display_response.ts
├── import type { DisplayResponse } from "data/viewer/utils/displays/utils/ts/frontend/types/display_response";
└── interface LayeredDisplayResponse extends DisplayResponse
    ├── slot_id  # common field
    ├── title    # common field
    ├── display_kind: "layered"  # common field
    ├── url        # common field
    ├── meta_info  # common field
    ├── base_display_response: DisplayResponse
    ├── aux_display_responses: DisplayResponse[]
    └── layer_class: "raster" | "spatial"  # backend-stamped (layered_display_response.layer_class); the frontend reads it instead of re-deriving the raster/spatial taxonomy
```

`data/viewer/utils/displays/utils/ts/frontend/layered_display_container.ts`

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
├── export function renderLayeredDisplay({ layeredDisplayResponse, initialCameraState, }: { layeredDisplayResponse: LayeredDisplayResponse; initialCameraState: CameraState | null; }): LeafVNode
│   ├── # Composes one layered display response into a shared spatial WebGL scene or a stacked raster DOM container per cell, routing on the backend-stamped layer_class.
│   ├── if layeredDisplayResponse.layer_class is "spatial"
│   │   ├── calls renderLayeredSpatialDisplay({ layeredDisplayResponse, initialCameraState })
│   │   └── return
│   ├── if layeredDisplayResponse.layer_class is "raster"
│   │   ├── calls renderLayeredRasterDisplay({ layeredDisplayResponse })
│   │   └── return
│   └── throw layered display response has an unknown layer class: ${JSON.stringify(layeredDisplayResponse.layer_class)}
├── function renderLayeredSpatialDisplay({ layeredDisplayResponse, initialCameraState, }: { layeredDisplayResponse: LayeredDisplayResponse; initialCameraState: CameraState | null; }): LeafVNode
│   ├── # Renders the base and aux spatial layers into one shared scene and camera, that camera owning the framing and the additive pick seam.
│   ├── () => [local]
│   │   ├── # The leaf's render: mounts the shared spatial context and returns its container.
│   │   ├── calls createSpatialDisplayScene({ initialCameraState })  # -> { container, scene, camera, renderer }
│   │   ├── calls createLayerObjects({ layeredDisplayResponse })     # -> layerObjects
│   │   ├── (object) => [local]
│   │   │   ├── # Per layer object: adds it to the one shared scene.
│   │   │   └── impls scene.add(object)
│   │   ├── impls layerObjects each added through that step
│   │   ├── calls createTrackballCameraControls({ container, camera, renderer, initialCameraState })  # -> controls, owned by the one shared camera
│   │   ├── calls _syncCameraState({ container, controls })
│   │   ├── calls attachThreeScenePickSeam({ container, camera, scenes: [scene] })
│   │   ├── calls renderLayeredSpatialScene({ scene, camera, renderer, controls })
│   │   └── return container
│   ├── impls leaf = the LeafVNode keyed by layeredDisplayResponse.slot_id, with empty props and that render
│   └── return leaf
├── function createLayerObjects({ layeredDisplayResponse, }: { layeredDisplayResponse: LayeredDisplayResponse; }): THREE.Object3D[]
│   ├── # Builds the THREE object for every layer by dispatching each layer's display response to its registry-resolved spatial renderer.
│   ├── impls layerObjects = []
│   ├── for each layer in the base display response followed by the aux display responses
│   │   ├── calls getSpatialLayerRenderer({ displayKind: layer.display_kind })  # -> layerRenderer
│   │   ├── calls layerRenderer({ displayResponse: layer })
│   │   └── impls layerObjects.push(that object)
│   └── return layerObjects
├── function renderLayeredSpatialScene({ scene, camera, renderer, controls, }: { scene: THREE.Scene; camera: THREE.PerspectiveCamera; renderer: THREE.WebGLRenderer; controls: ReturnType<typeof createTrackballCameraControls>; }): void
│   ├── # Drives the shared layered-scene render loop with the base camera's trackball controls.
│   └── calls startThreeSceneRenderLoop({ scene, camera, renderer, controls })
├── function renderLayeredRasterDisplay({ layeredDisplayResponse, }: { layeredDisplayResponse: LayeredDisplayResponse; }): LeafVNode
│   ├── # Stacks the base and aux raster layers full-bleed in one shared coordinate frame, keyed by slot_id.
│   ├── () => [local]
│   │   ├── # The leaf's render: materializes every layer cell and aligns the aux overlays to the base image's extent.
│   │   ├── impls container = document.createElement("div")
│   │   ├── impls container.className = "layered-display-container"
│   │   ├── impls container.style.position = "relative"
│   │   ├── impls container.style.width = "100%"
│   │   ├── impls container.style.height = "100%"
│   │   ├── impls auxCells = []
│   │   ├── impls layers = the base display response followed by the aux display responses
│   │   ├── (layer, layerIndex) => [local]
│   │   │   ├── # Per layer: materializes its full-bleed cell, hidden while it is an aux overlay awaiting alignment.
│   │   │   ├── calls getRasterLayerRenderer({ displayKind: layer.display_kind })  # -> layerRenderer
│   │   │   ├── impls cell = document.createElement("div")
│   │   │   ├── impls cell.style.position = "absolute"
│   │   │   ├── impls cell.style.inset = "0"
│   │   │   ├── impls cell.style.width = "100%"
│   │   │   ├── impls cell.style.height = "100%"
│   │   │   ├── calls layerRenderer({ displayResponse: layer })
│   │   │   ├── calls reconcileInto({ root: cell, virtualTree: that layer's vnode })
│   │   │   ├── impls container.appendChild(cell)
│   │   │   └── if layerIndex > 0
│   │   │       ├── impls cell.style.visibility = "hidden"  # hidden until alignAuxOverlays has set its viewBox, so no overlay flashes in the wrong coordinate space
│   │   │       └── impls auxCells.push(cell)
│   │   ├── impls layers each materialized through that step
│   │   ├── impls baseCell = container.firstElementChild
│   │   ├── impls baseImage = baseCell.querySelector("img")
│   │   ├── if baseImage is not null
│   │   │   ├── function alignAuxOverlays(): void [local]
│   │   │   │   ├── # Gives every aux overlay's svg the base image's natural pixel extent as its viewBox, then reveals it.
│   │   │   │   ├── calls _alignRasterFrustum({ baseImage })  # -> { width, height }
│   │   │   │   └── for each cell in auxCells
│   │   │   │       ├── impls svg = cell.querySelector("svg")
│   │   │   │       ├── if svg is not null
│   │   │   │       │   └── impls svg.setAttribute("viewBox", `0 0 ${width} ${height}`)
│   │   │   │       └── impls cell.style.visibility = "visible"  # revealed only now that its viewBox is the shared raster frustum
│   │   │   ├── if baseImage is already complete with a positive natural width
│   │   │   │   └── calls alignAuxOverlays()
│   │   │   └── else
│   │   │       └── impls baseImage.addEventListener("load", alignAuxOverlays)
│   │   └── return container
│   ├── impls leaf = the LeafVNode keyed by layeredDisplayResponse.slot_id, with empty props and that render
│   └── return leaf
├── function _syncCameraState({ container, controls, }: { container: HTMLDivElement; controls: ReturnType<typeof createTrackballCameraControls>; }): void
│   ├── # Publishes this cell's shared-camera pose now and re-publishes it on every controls change, so other cells can sync to it.
│   ├── calls _publishCameraState({ container, controls })
│   ├── () => [local]
│   │   ├── # The change listener: re-publishes the pose.
│   │   └── calls _publishCameraState({ container, controls })
│   └── impls controls.addEventListener("change", that listener)
├── function _publishCameraState({ container, controls, }: { container: HTMLDivElement; controls: ReturnType<typeof createTrackballCameraControls>; }): void
│   ├── # Publishes the controls' shared-camera state onto the container, as both a dataset attribute and a bubbling event, so the consumer can persist this cell's pose.
│   ├── impls cameraState = controls.getCameraState()
│   ├── if cameraState is null
│   │   └── return
│   ├── impls container.dataset.cameraState = the serialized cameraState
│   └── impls container.dispatchEvent(a bubbling "camera-pose-change" CustomEvent detailing cameraState)
└── function _alignRasterFrustum({ baseImage, }: { baseImage: HTMLImageElement; }): { width: number; height: number }
    ├── # Resolves the raster cell's shared frustum from the base image's natural pixel extent.
    └── return  # { width: baseImage.naturalWidth, height: baseImage.naturalHeight }
```


`data/viewer/utils/displays/utils/ts/frontend/layer_renderer_registry.ts`

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

`data/viewer/utils/displays/utils/ts/frontend/register_layer_renderers.ts`

```text
register_layer_renderers.ts
├── # Eager-imports every display modality's frontend apis module (Vite import.meta.glob) so each modality's module-load self-registration runs; new modalities are auto-discovered with no edit here.
└── impls import.meta.glob("data/viewer/utils/displays/**/ts/frontend/apis.ts", { eager: true })
```

`data/viewer/utils/displays/utils/ts/frontend/three_scene_helpers.ts`

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
│   ├── # Shared PerspectiveCamera factory for every TS atomic spatial display; the consumer-supplied initialCameraState is the single source of initial framing, with no lib-side fit-to-object.
│   ├── impls THREE.PerspectiveCamera(fov=DEFAULT_TRACKBALL_PERSPECTIVE_CAMERA_FOV, ...) at default aspect/near/far/position
│   ├── if initialCameraState is not null
│   │   └── impls overlays initialCameraState (every field — both intrinsics and extrinsics) onto the camera so first paint matches the source display  # impls-node-one-step:skip
│   └── return
├── function createThreeWebGLRenderer({ container }: { container: HTMLDivElement }): THREE.WebGLRenderer
│   ├── # Shared WebGL renderer factory for every TS atomic spatial display.
│   ├── impls renderer = new THREE.WebGLRenderer({ alpha: true })
│   ├── impls renderer.setClearColor(0x000000, 0)  # transparent canvas by default; an opaque backdrop is the consumer's CSS background-color on the marker
│   ├── impls canvas mounted inside the provided container
│   └── return
├── function createThreeScene(): THREE.Scene
│   ├── # Shared empty-scene factory used by every TS atomic spatial display; callers scene.add their own object(s).
│   ├── impls creates THREE.Scene; scene.background stays unset so the renderer's clear color is what gets visibly drawn
│   └── return
├── function attachThreeScenePickSeam({ container, camera, scenes }: { container: HTMLDivElement; camera: THREE.PerspectiveCamera; scenes: readonly THREE.Scene[] }): void
│   ├── # Installs a base-camera pickAt seam onto any spatial display container so a consumer can hit-test the given scenes via the camera without owning the camera, renderer, or scenes.
│   ├── impls raycaster = new THREE.Raycaster()
│   ├── function pickAt(clientX: number, clientY: number): THREE.Object3D | null [local]
│   │   ├── # The installed hit-test seam: maps a client point into the container's NDC and returns the first object the camera ray hits.
│   │   ├── impls rect = the container's bounding client rect
│   │   ├── if the rect is empty
│   │   │   └── return  # null: there is nothing to hit-test against
│   │   ├── impls ndcX = ((clientX - rect.left) / rect.width) * 2 - 1
│   │   ├── impls ndcY = -((clientY - rect.top) / rect.height) * 2 + 1  # client space is y-down, NDC is y-up
│   │   ├── impls ndc = new THREE.Vector2(ndcX, ndcY)
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
    │   │   ├── impls renderer.dispose(); renderer.forceContextLoss()
    │   │   └── return  # stop the loop without rescheduling
    │   ├── if controls is not null
    │   │   └── impls controls.update()
    │   ├── impls renderer.render(scene, camera)
    │   ├── if onAfterRender is provided
    │   │   └── impls onAfterRender()
    │   └── impls window.requestAnimationFrame(draw)
    └── impls window.requestAnimationFrame(draw)
```
