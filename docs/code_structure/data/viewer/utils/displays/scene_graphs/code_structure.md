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
├── function renderSceneGraphDisplay({ displayResponse, initialCameraState = null, nodeSize, edgeColor, edgeWidth, labelFontSize, labelColor, lockRoll = null }: { displayResponse: SceneGraphDisplayResponse; initialCameraState?: CameraState | null; nodeSize?: number; edgeColor?: string; edgeWidth?: number; labelFontSize?: number; labelColor?: string; lockRoll?: THREE.Vector3 | null }): LeafVNode
│   ├── # Renders a self-contained scene-graph display: baked node/edge geometry plus HTML label overlay projected per frame.
│   ├── calls createSpatialDisplayScene({ initialCameraState })
│   ├── calls createSceneGraphObject({ container, displayResponse, nodeSize, edgeColor, edgeWidth, labelFontSize, labelColor })   → { object, labels, labelOverlay }
│   ├── impls scene.add(object)
│   ├── calls createTrackballCameraControls({ container, camera, renderer, initialCameraState, lockRoll })
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
│   ├── impls payload = (await response.json()) as SceneGraphPayload  # cast unchecked
│   └── return payload
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
