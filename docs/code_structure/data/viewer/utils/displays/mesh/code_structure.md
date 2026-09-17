# Data Viewer Mesh Display Code Structure

## 1. Code structure trees

`data/viewer/utils/displays/mesh/dash/apis.py`

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

`data/viewer/utils/displays/mesh/dash/core_mesh_display.py`

```text
core_mesh_display.py
├── from typing import Any, Dict, Optional, Tuple
├── import plotly.graph_objects as go
├── from dash import dcc
├── from data.structures.three_d.mesh.mesh import Mesh
├── from data.viewer.utils.controls.camera.camera_controls.dash.trackball_camera_controls import create_dash_trackball_camera_controls
├── DEFAULT_MESH_COLOR = "#cccccc"  # uniform fallback color used when geometry has no texture AND has no per-vertex colors AND the caller does not supply mesh_color; lib-owned default, overridable
├── DEFAULT_MESH_OPACITY = 1.0      # opaque default applied when the caller does not supply mesh_opacity; lib-owned default, overridable
├── DEFAULT_MESH_SIDE = "double"    # fallback side mode for visibility under arbitrary camera framings when the caller does not supply mesh_side; lib-owned default, overridable
├── def create_dash_mesh_display(mesh: Any, mesh_color: Optional[str] = None, mesh_opacity: Optional[float] = None, mesh_side: Optional[str] = None, lock_roll: Optional[Tuple[float, float, float]] = None) -> dcc.Graph
│   ├── # Renders a Dash mesh display element with trackball camera controls; mesh_color, mesh_opacity, and mesh_side overrides are opt-in.
│   ├── calls create_dash_mesh_scene(mesh=mesh, mesh_color=mesh_color, mesh_opacity=mesh_opacity, mesh_side=mesh_side)
│   ├── calls create_dash_trackball_camera_controls(lock_roll=lock_roll)
│   ├── calls create_dash_mesh_component(scene=scene, controls=controls)   → display
│   └── return display  # the display element, carrying the roll-locked graph id when lock_roll is supplied
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
│   │   └── impls effective_color = mesh.texture.vertex_color
│   ├── else
│   │   └── impls effective_color = DEFAULT_MESH_COLOR
│   └── return
├── def _create_dash_uv_texture_map_mesh_scene(mesh: Any, mesh_color: Optional[str], effective_opacity: float, effective_side: str) -> go.Mesh3d
│   ├── # Builds the Plotly Mesh3d trace for a UV-texture-mapped mesh, resolving the effective color.
│   ├── if mesh_color is not None
│   │   └── impls effective_color = mesh_color
│   ├── elif mesh.texture carries a uv_texture_map
│   │   └── impls effective_color = sample(mesh.texture.uv_texture_map, mesh.texture.verts_uvs)
│   ├── else
│   │   └── impls effective_color = DEFAULT_MESH_COLOR
│   └── return
└── def create_dash_mesh_component(scene: go.Mesh3d, controls: Dict[str, Any]) -> dcc.Graph  # controls: the Plotly gl3d controls create_dash_trackball_camera_controls built
    ├── # Assembles the Dash component that hosts the Mesh3d scene under its trackball camera controls.
    ├── def _validate_inputs [local]
    │   └── assert isinstance(scene, go.Mesh3d), "..."
    ├── calls _validate_inputs()
    ├── impls display = dcc.Graph(figure=go.Figure(data=[scene], layout={"scene": controls["scene"]}))
    ├── if controls["graph_id"] is not None
    │   └── impls display.id = controls["graph_id"]  # the pattern-matching id the roll-lock callback holds this graph by
    └── return display  # the mesh display element
```

### Backend schemas

`data/viewer/utils/displays/mesh/ts/backend/schemas/display_response.py`

```text
display_response.py
├── from data.viewer.utils.displays.utils.ts.backend.schemas.display_response import DisplayResponse
├── class MeshDisplayResponse(DisplayResponse)
│   ├── slot_id       # common field
│   ├── title         # common field
│   ├── display_kind  # common field
│   ├── url           # common field
│   └── meta_info     # common field
├── class ColorMeshDisplayResponse(MeshDisplayResponse)
│   ├── slot_id  # common field
│   ├── title    # common field
│   ├── display_kind = "color_mesh"  # common field
│   ├── url        # common field
│   └── meta_info  # common field
├── class SegmentationMeshDisplayResponse(MeshDisplayResponse)
│   ├── slot_id  # common field
│   ├── title    # common field
│   ├── display_kind = "segmentation_mesh"  # common field
│   ├── url        # common field — the class-colorized mesh resource
│   └── meta_info  # common field
├── class HeatmapMeshDisplayResponse(MeshDisplayResponse)
│   ├── slot_id  # common field
│   ├── title    # common field
│   ├── display_kind = "heatmap_mesh"  # common field
│   ├── url        # common field — the heatmap-colorized mesh resource
│   └── meta_info  # common field
└── class SparseHeatmapMeshDisplayResponse(MeshDisplayResponse)
    ├── slot_id  # common field
    ├── title    # common field
    ├── display_kind = "sparse_heatmap_mesh"  # common field
    ├── url        # common field — the sparse heatmap wire resource: a shared-geometry reference plus the sparse (indices, values) delta
    └── meta_info  # common field
```

### Backend

`data/viewer/utils/displays/mesh/ts/backend/apis.py`

```text
apis.py
├── import json
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
│   ├── impls reads the (indices, values) delta and the geometry reference from input_path  # impls-node-one-step:skip
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
│   ├── # Writes the (indices, values) delta + geometry_url from input_path to output_path as the wire resource.
│   ├── impls geometry_url = the url of the mesh resource those indices index into, read from input_path
│   ├── impls indices, values = the non-default entries read from input_path
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
└── def _build_sparse_heatmap_mesh_meta_info
    ├── # Builds scalar-range + non-zero-count metadata from the input sparse arrays.
    ├── impls stores values min/max and number of non-zero entries  # impls-node-one-step:skip
    └── return
```

`data/viewer/utils/displays/mesh/ts/backend/core_mesh_display.py`

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

### Frontend

`data/viewer/utils/displays/mesh/ts/frontend/types/display_response.ts`

```text
display_response.ts
├── import type { DisplayResponse } from "data/viewer/utils/displays/utils/ts/frontend/types/display_response";
├── interface MeshDisplayResponse extends DisplayResponse
│   ├── slot_id       # common field
│   ├── title         # common field
│   ├── display_kind  # common field
│   ├── url           # common field
│   └── meta_info     # common field
├── interface ColorMeshDisplayResponse extends MeshDisplayResponse
│   ├── slot_id  # common field
│   ├── title    # common field
│   ├── display_kind = "color_mesh"  # common field
│   ├── url        # common field
│   └── meta_info  # common field
├── interface SegmentationMeshDisplayResponse extends MeshDisplayResponse
│   ├── slot_id  # common field
│   ├── title    # common field
│   ├── display_kind = "segmentation_mesh"  # common field
│   ├── url        # common field — the class-colorized mesh resource
│   └── meta_info  # common field
├── interface HeatmapMeshDisplayResponse extends MeshDisplayResponse
│   ├── slot_id  # common field
│   ├── title    # common field
│   ├── display_kind = "heatmap_mesh"  # common field
│   ├── url        # common field — the heatmap-colorized mesh resource
│   └── meta_info  # common field
└── interface SparseHeatmapMeshDisplayResponse extends MeshDisplayResponse
    ├── slot_id  # common field
    ├── title    # common field
    ├── display_kind = "sparse_heatmap_mesh"  # common field
    ├── url        # common field — the sparse heatmap wire resource: a shared-geometry reference plus the sparse (indices, values) delta
    └── meta_info  # common field
```

`data/viewer/utils/displays/mesh/ts/frontend/core_mesh_display.ts`

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
├── function renderMeshDisplay({ displayResponse, initialCameraState = null, meshColor, meshOpacity, meshSide, lockRoll = null }: { displayResponse: MeshDisplayResponse; initialCameraState?: CameraState | null; meshColor?: string; meshOpacity?: number; meshSide?: THREE.Side; lockRoll?: THREE.Vector3 | null }): LeafVNode
│   ├── # Renders a self-contained mesh display element initialized at initialCameraState.
│   ├── calls createSpatialDisplayScene({ initialCameraState })
│   ├── calls createMeshObject({ displayResponse, meshColor, meshOpacity, meshSide })   → object
│   ├── impls scene.add(object)
│   ├── calls createTrackballCameraControls({ container, camera, renderer, initialCameraState, lockRoll })
│   ├── calls renderMeshScene({ scene, camera, renderer, controls })
│   └── return LeafVNode keyed by displayResponse.url
├── function createMeshObject({ displayResponse, meshColor, meshOpacity, meshSide }: { displayResponse: MeshDisplayResponse; meshColor?: string; meshOpacity?: number; meshSide?: THREE.Side }): THREE.Object3D
│   ├── # Part-B: returns a THREE.Group for the mesh, populated with the THREE.Mesh once the async payload load resolves.
│   ├── impls group = new THREE.Group()
│   ├── impls loadMeshPayload({ displayResponse }).then(payload => group.add(createThreeMesh({ payload, displayResponse, meshColor, meshOpacity, meshSide })))
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
│   └── return new THREE.Mesh(geometry, material)  # no post-construction mutation of mesh
└── function renderMeshScene({ scene, camera, renderer, controls }: { scene: THREE.Scene; camera: THREE.PerspectiveCamera; renderer: THREE.WebGLRenderer; controls: ReturnType<typeof createTrackballCameraControls>; }): void
    ├── # Drives the mesh render loop with the supplied trackball controls.
    ├── calls startThreeSceneRenderLoop({ scene, camera, renderer, controls })
    └── return
```

`data/viewer/utils/displays/mesh/ts/frontend/apis.ts`

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
