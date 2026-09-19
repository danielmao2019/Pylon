# Data Viewer Scene Graphs Display Code Structure

## 1. Code structure trees

`data/viewer/utils/displays/scene_graphs/dash/scene_graph_display.py`

```text
scene_graph_display.py
├── from typing import Dict, List
├── from dash import html
└── def create_scene_graph_display(rows: List[Dict[str, str]]) -> html.Pre
    ├── # Builds the Dash scene-graph display from the scene-graph preview rows.
    ├── impls assert isinstance(rows, list)
    ├── impls display = html.Pre(str(rows), className="json-preview")
    └── return display
```

### Backend schemas

`data/viewer/utils/displays/scene_graphs/ts/backend/schemas/display_response.py`

```text
display_response.py
├── from data.viewer.utils.displays.utils.ts.backend.schemas.display_response import DisplayResponse
└── class SceneGraphDisplayResponse(DisplayResponse)
    ├── slot_id  # common field
    ├── title    # common field
    ├── display_kind = "scene_graph"  # common field
    ├── url        # common field; serves the scene-graph payload (no leaked encoding)
    └── meta_info  # common field
```

### Backend

`data/viewer/utils/displays/scene_graphs/ts/backend/scene_graph_display.py`

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
├── def estimate_scene_scale
│   ├── # Returns the world-units diagonal of the union of object positions, camera trajectory, and graph_nodes positions.
│   └── return
├── def bake_scene_graph_geometry
│   ├── # Bakes sphere-sampled nodes + line-sampled edges into the scene-graph geometry asset.
│   ├── calls sample_node_spheres
│   ├── calls sample_edge_lines
│   └── return
├── def bake_scene_graph_labels
│   ├── # Bakes per-object-node labels (text, position, color, class identity, frequency) offset above each position by scene_scale.
│   └── return
├── def sample_node_spheres
│   ├── # Samples each graph node into a sphere-shaped point patch, with radius derived from node_type and scene_scale, colored by node.color.
│   └── return
└── def sample_edge_lines
    ├── # Samples each graph edge into a densely-sampled line from source.position to target.position, colored by edge color.
    └── return
```

### Frontend

`data/viewer/utils/displays/scene_graphs/ts/frontend/types/display_response.ts`

```text
display_response.ts
├── import type { DisplayResponse } from "data/viewer/utils/displays/utils/ts/frontend/types/display_response";
└── interface SceneGraphDisplayResponse extends DisplayResponse
    ├── slot_id  # common field
    ├── title    # common field
    ├── display_kind = "scene_graph"  # common field
    ├── url        # common field; serves the scene-graph payload (no leaked encoding)
    └── meta_info  # common field
```

`data/viewer/utils/displays/scene_graphs/ts/frontend/scene_graph_display.ts`

```text
scene_graph_display.ts
├── import * as THREE from "three";
├── import { createTrackballCameraControls, type ThreeTrackballCameraControls } from "data/viewer/utils/controls/camera/camera_controls/ts/frontend/trackball_camera_controls";
├── import type { CameraState } from "data/viewer/utils/controls/camera/camera_state/ts/frontend/types";
├── import { createSpatialDisplayScene, startThreeSceneRenderLoop } from "data/viewer/utils/displays/utils/ts/frontend/three_scene_helpers";
├── import type { LeafVNode } from "web/reconcile/reconcile";
├── import type { SceneGraphDisplayResponse } from "./types/display_response";
├── export const DEFAULT_NODE_SIZE = 0.02         # heuristic world-space size for node markers when the caller supplies no nodeSize; lib-owned default, overridable
├── export const DEFAULT_EDGE_COLOR = "#888888"   # neutral gray for edge lines when neither the payload nor the caller carries an edge color; lib-owned default, overridable
├── export const DEFAULT_EDGE_WIDTH = 1.0         # line width for edges when the caller supplies no edgeWidth; lib-owned default, overridable
├── export const DEFAULT_LABEL_FONT_SIZE = 12     # px font size for overlay labels when the caller supplies no labelFontSize; lib-owned default, overridable
├── export const DEFAULT_LABEL_COLOR = "#000000"  # text color for overlay labels when the caller supplies no labelColor; lib-owned default, overridable
├── interface SceneGraphLabelEntry
│   ├── # One per-node label of the payload: the text and the world position the per-frame projection drives it from.
│   ├── text: string
│   └── position: { x: number; y: number; z: number; }
├── interface SceneGraphPayload
│   ├── # Minimal payload the scene-graph display consumes: node and edge positions with optional colors, plus the label entries.
│   ├── nodePositions: number[]
│   ├── nodeColors?: number[]
│   ├── edgePositions: number[]
│   ├── edgeColors?: number[]
│   └── labels: SceneGraphLabelEntry[]
├── export function renderSceneGraphDisplay({ displayResponse, initialCameraState = null, nodeSize, edgeColor, edgeWidth, labelFontSize, labelColor, }: { displayResponse: SceneGraphDisplayResponse; initialCameraState?: CameraState | null; nodeSize?: number; edgeColor?: string; edgeWidth?: number; labelFontSize?: number; labelColor?: string; }): LeafVNode
│   ├── # Renders a self-contained scene-graph display: baked node and edge geometry plus an HTML label overlay projected per frame.
│   ├── () => [local]
│   │   ├── # The leaf's render: mounts the scene-graph scene and returns its container.
│   │   ├── calls createSpatialDisplayScene({ initialCameraState })  # -> { container, scene, camera, renderer }
│   │   ├── calls createSceneGraphObject({ container, displayResponse, nodeSize, edgeColor, edgeWidth, labelFontSize, labelColor })  # -> { object, labels, labelOverlay }
│   │   ├── impls scene.add(object)
│   │   ├── calls createTrackballCameraControls({ container, camera, renderer, initialCameraState })  # -> controls
│   │   ├── calls renderSceneGraphScene({ scene, camera, renderer, controls, labels, labelOverlay, labelFontSize, labelColor })
│   │   └── return container
│   ├── impls leaf = the LeafVNode keyed by displayResponse.url or `scene_graph:${displayResponse.slot_id}`, with empty props and that render
│   └── return leaf
├── function createSceneGraphObject({ container, displayResponse, nodeSize, edgeColor, edgeWidth, labelFontSize, labelColor, }: { container: HTMLDivElement; displayResponse: SceneGraphDisplayResponse; nodeSize?: number; edgeColor?: string; edgeWidth?: number; labelFontSize?: number; labelColor?: string; }): { object: THREE.Object3D; labels: object[]; labelOverlay: HTMLDivElement }
│   ├── # Part-B: builds the HTML label overlay and returns a THREE.Group and mutable labels array, both filled once the async payload load resolves.
│   ├── calls createThreeSceneGraphLabelOverlay({ container, labelFontSize, labelColor })  # -> labelOverlay
│   ├── impls group = new THREE.Group()
│   ├── impls labels = []  # initially empty, mutated on async resolve so the per-frame projection sees the filled list
│   ├── calls loadSceneGraphPayload({ displayResponse })
│   ├── (payload) => [local]
│   │   ├── # On resolve: builds the points and labels from the payload into the already-returned group and array.
│   │   ├── calls createThreeSceneGraphPoints({ payload, nodeSize, edgeColor, edgeWidth })  # -> built
│   │   ├── impls group.add(built.points)
│   │   └── impls labels.push(...built.labels)
│   ├── (error) => [local]
│   │   ├── # On rejection: throws a new Error carrying the underlying message.
│   │   ├── if error is an Error
│   │   │   └── impls error.message
│   │   ├── else
│   │   │   └── impls String(error)
│   │   ├── impls message = that text
│   │   └── throw unable to load scene graph: ${message}
│   ├── impls the payload load, chained through that resolve step and that rejection step
│   └── return { object: group, labels, labelOverlay }
├── function createThreeSceneGraphLabelOverlay({ container, labelFontSize, labelColor, }: { container: HTMLDivElement; labelFontSize?: number; labelColor?: string; }): HTMLDivElement
│   ├── # Builds the absolutely-positioned HTML overlay layered above the canvas, carrying the label font size and color as its defaults.
│   ├── impls effectiveLabelFontSize = labelFontSize ?? DEFAULT_LABEL_FONT_SIZE
│   ├── impls effectiveLabelColor = labelColor ?? DEFAULT_LABEL_COLOR
│   ├── impls overlay = document.createElement("div")
│   ├── impls overlay.style.position = "absolute"
│   ├── impls overlay.style.inset = "0"
│   ├── impls overlay.style.width = "100%"
│   ├── impls overlay.style.height = "100%"
│   ├── impls overlay.style.overflow = "hidden"
│   ├── impls overlay.style.pointerEvents = "none"
│   ├── impls overlay.style.fontSize = `${effectiveLabelFontSize}px`
│   ├── impls overlay.style.color = effectiveLabelColor
│   ├── impls container.append(overlay)
│   └── return overlay
├── async function loadSceneGraphPayload({ displayResponse, }: { displayResponse: SceneGraphDisplayResponse; }): Promise<SceneGraphPayload>
│   ├── # Async-loads the scene-graph payload served at displayResponse.url.
│   ├── if displayResponse.url is null
│   │   └── throw new Error("scene graph display response url is null")
│   ├── impls response = await fetch(displayResponse.url)
│   ├── if the response is not ok
│   │   └── throw new Error(`unable to load scene graph: HTTP ${response.status}`)
│   └── return  # the awaited response body parsed as JSON, cast unchecked to SceneGraphPayload
├── function createThreeSceneGraphPoints({ payload, nodeSize, edgeColor, edgeWidth, }: { payload: SceneGraphPayload; nodeSize?: number; edgeColor?: string; edgeWidth?: number; }): { points: THREE.Points; labels: object[] }
│   ├── # Sync-builds the THREE.Points, its edge line set, and the per-frame label data from a loaded payload.
│   ├── impls effectiveNodeSize = nodeSize ?? DEFAULT_NODE_SIZE
│   ├── impls effectiveEdgeWidth = edgeWidth ?? DEFAULT_EDGE_WIDTH
│   ├── impls let useEdgeVertexColors: boolean
│   ├── impls let effectiveEdgeColor: string | undefined
│   ├── if edgeColor is supplied
│   │   ├── impls useEdgeVertexColors = false
│   │   └── impls effectiveEdgeColor = edgeColor
│   ├── else if the payload carries per-edge colors
│   │   ├── impls useEdgeVertexColors = true
│   │   └── impls effectiveEdgeColor = undefined
│   ├── else
│   │   ├── impls useEdgeVertexColors = false
│   │   └── impls effectiveEdgeColor = DEFAULT_EDGE_COLOR
│   ├── impls nodeGeometry = new THREE.BufferGeometry()
│   ├── impls nodeGeometry.setAttribute("position", the 3-component buffer over payload.nodePositions)
│   ├── impls useNodeVertexColors = whether the payload carries per-node colors
│   ├── if useNodeVertexColors
│   │   └── impls nodeGeometry.setAttribute("color", the 3-component buffer over payload.nodeColors)
│   ├── impls nodeMaterial = new THREE.PointsMaterial(vertexColors useNodeVertexColors, size effectiveNodeSize)
│   ├── impls points = new THREE.Points(nodeGeometry, nodeMaterial)
│   ├── impls edgeGeometry = new THREE.BufferGeometry()
│   ├── impls edgeGeometry.setAttribute("position", the 3-component buffer over payload.edgePositions)
│   ├── if useEdgeVertexColors
│   │   └── impls edgeGeometry.setAttribute("color", the 3-component buffer over payload.edgeColors)
│   ├── if effectiveEdgeColor is supplied
│   │   └── impls { color: effectiveEdgeColor }
│   ├── else
│   │   └── impls {}
│   ├── impls edgeMaterial = new THREE.LineBasicMaterial(vertexColors useEdgeVertexColors, linewidth effectiveEdgeWidth, spread with that color entry)
│   ├── impls points.add(new THREE.LineSegments(edgeGeometry, edgeMaterial))
│   ├── (entry) => [local]
│   │   ├── # Per label entry: builds its absolutely-positioned overlay node and pairs it with its world position.
│   │   ├── impls node = document.createElement("div")
│   │   ├── impls node.style.position = "absolute"
│   │   ├── impls node.style.whiteSpace = "nowrap"
│   │   ├── impls node.textContent = entry.text
│   │   └── return  # { node, position: new THREE.Vector3(entry.position.x, entry.position.y, entry.position.z) }
│   ├── impls labels = payload.labels mapped through that step
│   └── return { points, labels }
├── function renderSceneGraphScene({ scene, camera, renderer, controls, labels, labelOverlay, labelFontSize, labelColor, }: { scene: THREE.Scene; camera: THREE.PerspectiveCamera; renderer: THREE.WebGLRenderer; controls: ThreeTrackballCameraControls; labels: object[]; labelOverlay: HTMLDivElement; labelFontSize?: number; labelColor?: string; }): void
│   ├── # Drives the render loop, projecting the labels onto the overlay after each frame.
│   ├── () => [local]
│   │   ├── # The loop's onAfterRender step.
│   │   └── calls _projectLabelsOntoOverlay({ camera, labels, labelOverlay, labelFontSize, labelColor })
│   └── calls startThreeSceneRenderLoop({ scene, camera, renderer, controls, onAfterRender: that step })
└── function _projectLabelsOntoOverlay({ camera, labels, labelOverlay, labelFontSize, labelColor, }: { camera: THREE.PerspectiveCamera; labels: object[]; labelOverlay: HTMLDivElement; labelFontSize?: number; labelColor?: string; }): void
    ├── # Per-frame step: projects each label's world position into overlay-pixel coordinates and culls the offscreen ones.
    ├── impls effectiveLabelFontSize = labelFontSize ?? DEFAULT_LABEL_FONT_SIZE
    ├── impls effectiveLabelColor = labelColor ?? DEFAULT_LABEL_COLOR
    ├── impls width = max(1, labelOverlay.clientWidth or 1)
    ├── impls height = max(1, labelOverlay.clientHeight or 1)
    └── for each label in labels
        ├── impls (node, position) = the label, read as its overlay node and world position
        ├── if node.parentElement is not labelOverlay
        │   └── impls labelOverlay.append(node)
        ├── impls projected = position.clone().project(camera)
        ├── impls offscreen = projected.z > 1 or projected.x outside [-1, 1] or projected.y outside [-1, 1]
        ├── if offscreen
        │   ├── impls node.style.display = "none"
        │   └── continue
        ├── impls left = ((projected.x + 1) / 2) * width
        ├── impls top = ((1 - projected.y) / 2) * height
        ├── impls node.style.display = "block"
        ├── impls node.style.left = `${left}px`
        ├── impls node.style.top = `${top}px`
        ├── impls node.style.fontSize = `${effectiveLabelFontSize}px`
        └── impls node.style.color = effectiveLabelColor
```
