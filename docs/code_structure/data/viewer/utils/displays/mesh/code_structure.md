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
├── import base64
├── import io
├── import json
├── from functools import lru_cache
├── from pathlib import Path
├── from typing import Any, Dict, List, Optional, Tuple, Union
├── import numpy as np
├── import plotly.graph_objects as go
├── import torch
├── from dash import dcc, html
├── from PIL import Image
├── from data.structures.three_d.mesh.mesh import Mesh
├── from data.structures.three_d.mesh.texture.mesh_texture_uv_texture_map import MeshTextureUVTextureMap
├── from data.structures.three_d.mesh.texture.mesh_texture_vertex_color import MeshTextureVertexColor
├── from data.viewer.utils.controls.camera.camera_controls.dash.trackball_camera_controls import create_dash_trackball_camera_controls
├── from data.viewer.utils.controls.camera.camera_sync.threejs import build_threejs_camera_sync_script
├── DEFAULT_MESH_COLOR = "#cccccc"  # lib-owned default mesh color, overridable
├── DEFAULT_MESH_OPACITY = 1.0      # opaque default when the caller supplies no mesh_opacity; lib-owned default, overridable
├── DEFAULT_MESH_SIDE = "double"    # the side mode effective_side falls back to; lib-owned default, overridable
├── MODULE_DIR = Path(__file__).resolve().parent
├── TEXTURED_MESH_VIEWER_SCRIPT_PATH = MODULE_DIR / "mesh_display_textured_viewer.js"
├── PLOTLY_MESH_DEFAULT_CAMERA_EYE_Z = 2.4                                # the z the Plotly mesh scene's camera eye sits at
├── THREEJS_MESH_DEFAULT_CAMERA_EYE_Z = PLOTLY_MESH_DEFAULT_CAMERA_EYE_Z  # the three.js viewer frames the mesh from the same eye distance as the Plotly scene
├── PLOTLY_MESH_CAMERA_UIREVISION = "mesh-display-camera"                 # one uirevision across mesh figures, so a redraw keeps the user's camera
├── MESH_VIEW_BOUNDS_KEYS = {"center", "camera_coordinate_scale", "half_span", "max_span", "axis_ranges"}                    # the shared mesh-view-bounds contract
├── TEXTURED_MESH_IFRAME_STYLE: Dict[str, str] = {"width": "100%", "height": "100%", "minHeight": "0", "border": "1px solid #dce5f0", "borderRadius": "12px", "overflow": "hidden", "backgroundColor": "#f7fafc"}
├── def create_mesh_display(mesh: Mesh, title: str, component_id: Optional[str] = None, camera_sync_group: Optional[str] = None) -> Union[go.Figure, html.Iframe]
│   ├── # Creates one display component from a generic mesh container, routing on the texture it carries.
│   ├── def _normalize_inputs() -> Tuple[str, str, Optional[str]] [local]
│   │   ├── impls normalized_title = title.strip()
│   │   ├── calls _normalize_component_id(component_id=component_id, title=normalized_title)  # -> normalized_component_id
│   │   ├── impls normalized_camera_sync_group = None
│   │   ├── if camera_sync_group is not None
│   │   │   └── impls normalized_camera_sync_group = camera_sync_group.strip()
│   │   ├── impls normalized_inputs = (normalized_title, normalized_component_id, normalized_camera_sync_group)
│   │   └── return normalized_inputs
│   ├── calls _normalize_inputs()  # -> (normalized_title, normalized_component_id, normalized_camera_sync_group)
│   ├── if mesh.texture is a MeshTextureVertexColor
│   │   ├── calls _create_vertex_color_mesh_display(mesh=mesh, title=normalized_title)
│   │   └── return
│   ├── calls _create_uv_texture_mesh_display(mesh=mesh, title=normalized_title, component_id=normalized_component_id, camera_sync_group=normalized_camera_sync_group)
│   └── return
├── def _create_vertex_color_mesh_display(mesh: Mesh, title: str) -> go.Figure
│   ├── # Creates one Plotly mesh figure from per-vertex colors.
│   ├── for vertex_rgb in vertex_colors
│   │   └── calls _rgb_to_css_color(rgb_values=vertex_rgb)
│   ├── impls figure = a go.Figure holding one Mesh3d over the verts and faces, vertex-colored by those CSS colors, smooth-shaded under the fixed lighting and light position, hover skipped, named title  # impls-node-one-step:skip — one constructor call's arguments
│   ├── calls _apply_mesh_layout(figure=figure, title=title, mesh_view_bounds=mesh_view_bounds)
│   └── return figure
├── def _create_uv_texture_mesh_display(mesh: Mesh, title: str, component_id: str, camera_sync_group: Optional[str]) -> html.Iframe
│   ├── # Creates one Three.js iframe from a UV texture map.
│   ├── def _normalize_inputs() -> Tuple[List[float], List[float], str, Dict[str, object]] [local]
│   │   ├── calls mesh.to(uv_convention="obj")  # -> display_mesh, in the OBJ UV convention the viewer reads
│   │   ├── impls normalized_verts = display_mesh.verts.detach().cpu().numpy()
│   │   ├── impls normalized_faces = display_mesh.faces.detach().cpu().numpy()
│   │   ├── impls normalized_verts_uvs = display_mesh.texture.verts_uvs.detach().cpu().numpy()
│   │   ├── impls normalized_faces_uvs = display_mesh.texture.faces_uvs.detach().cpu().numpy()
│   │   ├── calls build_mesh_view_bounds(verts=display_mesh.verts)  # -> normalized_mesh_view_bounds
│   │   ├── calls _build_textured_triangle_buffers(verts=normalized_verts, faces=normalized_faces, verts_uvs=normalized_verts_uvs, faces_uvs=normalized_faces_uvs)  # -> (normalized_triangle_positions, normalized_triangle_uvs)
│   │   ├── calls _normalize_texture_map_to_uint8(texture_map=display_mesh.texture.uv_texture_map)  # -> normalized_texture_map
│   │   ├── calls _build_texture_data_url(texture_map=normalized_texture_map)                       # -> normalized_texture_data_url
│   │   ├── impls normalized_inputs = (the triangle positions and uvs as lists, normalized_texture_data_url, normalized_mesh_view_bounds)  # impls-node-one-step:skip — one tuple's entries
│   │   └── return normalized_inputs
│   ├── calls _normalize_inputs()  # -> (triangle_position_values, triangle_uv_values, texture_data_url, mesh_view_bounds)
│   ├── calls _build_textured_mesh_html(title=title, position_values=triangle_position_values, uv_values=triangle_uv_values, texture_data_url=texture_data_url, mesh_view_bounds=mesh_view_bounds, viewer_id=component_id, camera_sync_group=camera_sync_group)  # -> iframe_html
│   ├── impls iframe_attributes = {}
│   ├── if camera_sync_group is not None
│   │   └── impls iframe_attributes = {"data-camera-sync-group": camera_sync_group, "data-camera-sync-viewer-id": component_id}
│   ├── impls iframe = the html.Iframe id'd by component_id, serving iframe_html under TEXTURED_MESH_IFRAME_STYLE and those attributes  # impls-node-one-step:skip — one constructor call's arguments
│   └── return iframe
├── def _build_textured_mesh_html(title: str, position_values: List[float], uv_values: List[float], texture_data_url: str, mesh_view_bounds: Dict[str, object], viewer_id: str, camera_sync_group: Optional[str]) -> str
│   ├── # Builds the iframe HTML that renders one textured mesh with Three.js.
│   ├── calls _load_javascript_template(template_path=TEXTURED_MESH_VIEWER_SCRIPT_PATH)  # -> viewer_script_template
│   ├── def _build_viewer_script(camera_sync_script: str) -> str [local]
│   │   ├── # Builds one textured-mesh viewer script with the given camera sync.
│   │   ├── assert camera_sync_script is a str  # reporting its type
│   │   ├── impls filled_viewer_script = viewer_script_template with its position, uv, texture, bounds, camera-sync, default eye-z and viewer-id placeholders filled  # impls-node-one-step:skip — names filled placeholders
│   │   └── return filled_viewer_script
│   ├── calls _build_viewer_script(camera_sync_script="")  # -> viewer_script_without_camera_sync
│   ├── calls create_dash_trackball_camera_controls(renderer_controls=viewer_script_without_camera_sync)  # asserts the viewer's own controls are a free trackball
│   ├── calls build_threejs_camera_sync_script(viewer_id=viewer_id, camera_sync_group=camera_sync_group)  # -> camera_sync_script
│   ├── calls _build_viewer_script(camera_sync_script=camera_sync_script)                                 # -> viewer_script
│   ├── calls build_threejs_viewer_html(title=title, viewer_script=viewer_script)
│   └── return
├── @lru_cache(maxsize=None) def _load_javascript_template(template_path: Path) -> str
│   ├── # Loads one JavaScript template file from disk, once per path.
│   ├── impls template_text = that file's text, read as utf-8
│   └── return template_text
├── def build_threejs_viewer_html(title: str, viewer_script: str, extra_script_urls: Optional[List[str]] = None) -> str
│   ├── # Builds one generic Three.js iframe HTML document around a viewer script.
│   ├── if extra_script_urls is None
│   │   └── impls []
│   ├── else
│   │   └── impls extra_script_urls
│   ├── impls normalized_extra_script_urls = that value
│   ├── for script_url in normalized_extra_script_urls
│   │   └── impls that url as a script tag
│   ├── impls extra_script_tags = those tags, one per line
│   ├── impls extra_script_tags_block = ""
│   ├── if extra_script_tags is non-empty
│   │   └── impls extra_script_tags_block = extra_script_tags on its own line
│   ├── impls viewer_html = the HTML document: the title, the three.js script and extra_script_tags_block, the full-bleed styles, an empty div id'd mesh-root, and viewer_script in a script element after it  # impls-node-one-step:skip — one template's parts
│   └── return viewer_html
├── def _apply_mesh_layout(figure: go.Figure, title: str, mesh_view_bounds: Dict[str, object]) -> None
│   ├── # Applies the shared 3D mesh layout styling to one Plotly figure, in place.
│   ├── impls axis_ranges = mesh_view_bounds["axis_ranges"]
│   ├── assert axis_ranges is a dict  # reporting axis_ranges
│   └── impls figure.update_layout(meta {"meshViewBounds": mesh_view_bounds}, title, the margins and backgrounds, the data-aspect scene with hidden axes over those axis ranges, framed from PLOTLY_MESH_DEFAULT_CAMERA_EYE_Z with +Y up, no legend, uirevision PLOTLY_MESH_CAMERA_UIREVISION)  # impls-node-one-step:skip — one call's arguments
├── def _normalize_component_id(component_id: Optional[str], title: str) -> str
│   ├── # Normalizes one optional component id, falling back to one derived from the title.
│   └── return
├── def build_mesh_view_bounds(verts: torch.Tensor) -> Dict[str, object]
│   ├── # Builds one renderer framing summary from raw mesh verts.
│   ├── def _normalize_inputs() -> Dict[str, object] [local]
│   │   ├── impls min_corner = the per-axis minimum of verts, keeping the dim
│   │   ├── impls max_corner = the per-axis maximum of verts, keeping the dim
│   │   ├── impls bounds_center = the midpoint of min_corner and max_corner  # impls-node-one-step:skip — names two operands
│   │   ├── impls bounds_extent = max_corner - min_corner
│   │   ├── impls max_extent = the largest axis extent, as a float
│   │   ├── assert max_extent > 0.0  # "Expected mesh bounds to have a positive extent before building view bounds", reporting bounds_extent
│   │   ├── calls _compute_camera_coordinate_scale(bounds_extent=bounds_extent)  # -> camera_coordinate_scale
│   │   ├── impls half_span = max_extent / 2.0
│   │   ├── impls center_values = bounds_center flattened onto the cpu as a list
│   │   ├── impls min_corner_values = min_corner flattened onto the cpu as a list
│   │   ├── impls max_corner_values = max_corner flattened onto the cpu as a list
│   │   ├── impls mesh_view_bounds = { center, camera_coordinate_scale, half_span, max_span: max_extent, axis_ranges: the per-axis [min, max] }
│   │   └── return mesh_view_bounds
│   ├── calls _normalize_inputs()
│   └── return
├── def _compute_camera_coordinate_scale(bounds_extent: torch.Tensor) -> float
│   ├── # Computes one shared camera-coordinate scale from a mesh's axis extents.
│   ├── impls flattened_extent = bounds_extent flattened onto the cpu as float64
│   ├── impls positive_extent = the entries of flattened_extent above 0
│   ├── assert positive_extent is non-empty  # reporting flattened_extent
│   ├── impls camera_coordinate_scale = the geometric mean of positive_extent
│   └── return camera_coordinate_scale
├── def validate_mesh_view_bounds(mesh_view_bounds: Dict[str, object]) -> None
│   ├── # Validates one shared mesh-view-bounds payload against the contract.
│   ├── assert mesh_view_bounds is a dict                  # reporting its type
│   └── assert its keys are exactly MESH_VIEW_BOUNDS_KEYS  # reporting mesh_view_bounds
├── def create_dash_mesh_display(mesh: Any, mesh_color: Optional[str] = None, mesh_opacity: Optional[float] = None, mesh_side: Optional[str] = None) -> dcc.Graph
│   ├── # Renders a Dash mesh display element; the mesh_color, mesh_opacity and mesh_side overrides are opt-in.
│   ├── assert mesh is a Mesh                   # reporting its type
│   ├── assert mesh_color is None or a str      # reporting its type
│   ├── assert mesh_opacity is None or numeric  # reporting its type
│   ├── assert mesh_side is None or a str       # reporting its type
│   ├── calls create_dash_mesh_scene(mesh=mesh, mesh_color=mesh_color, mesh_opacity=mesh_opacity, mesh_side=mesh_side)  # -> scene
│   ├── impls controls = create_dash_trackball_camera_controls
│   ├── calls create_dash_mesh_component(scene=scene, controls=controls)
│   └── return
├── def create_dash_mesh_scene(mesh: Any, mesh_color: Optional[str] = None, mesh_opacity: Optional[float] = None, mesh_side: Optional[str] = None) -> go.Mesh3d
│   ├── # Sync-builds the Plotly Mesh3d trace from the mesh, routing on the texture it carries.
│   ├── assert mesh is a Mesh                   # reporting its type
│   ├── assert mesh_color is None or a str      # reporting its type
│   ├── assert mesh_opacity is None or numeric  # reporting its type
│   ├── assert mesh_side is None or a str       # reporting its type
│   ├── if mesh_opacity is not None
│   │   └── impls mesh_opacity
│   ├── else
│   │   └── impls DEFAULT_MESH_OPACITY
│   ├── impls effective_opacity = that value
│   ├── if mesh_side is not None
│   │   └── impls mesh_side
│   ├── else
│   │   └── impls DEFAULT_MESH_SIDE
│   ├── impls effective_side = that value
│   ├── if mesh.texture is a MeshTextureVertexColor
│   │   ├── calls _create_dash_vertex_color_mesh_scene(mesh=mesh, mesh_color=mesh_color, effective_opacity=effective_opacity, effective_side=effective_side)
│   │   └── return
│   ├── if mesh.texture is a MeshTextureUVTextureMap
│   │   ├── calls _create_dash_uv_texture_map_mesh_scene(mesh=mesh, mesh_color=mesh_color, effective_opacity=effective_opacity, effective_side=effective_side)
│   │   └── return
│   └── raise ValueError  # the mesh carries neither supported texture
├── def _create_dash_vertex_color_mesh_scene(mesh: Any, mesh_color: Optional[str], effective_opacity: float, effective_side: str) -> go.Mesh3d
│   ├── # Sync-builds the Plotly Mesh3d trace from a vertex-colored mesh.
│   ├── assert mesh is a Mesh                            # reporting its type
│   ├── assert mesh.texture is a MeshTextureVertexColor  # reporting its type
│   ├── assert mesh_color is None or a str               # reporting its type
│   ├── assert effective_opacity is numeric              # reporting its type
│   ├── assert effective_side is a str                   # reporting its type
│   ├── impls verts_np = mesh.verts.detach().cpu().numpy()
│   ├── impls faces_np = mesh.faces.detach().cpu().numpy()
│   ├── if mesh_color is not None
│   │   └── impls effective_color = mesh_color
│   ├── else
│   │   ├── calls _normalize_rgb_tensor_to_uint8(rgb_values=mesh.texture.vertex_color)  # -> vertex_colors
│   │   ├── for vertex_rgb in vertex_colors
│   │   │   └── calls _rgb_to_css_color(rgb_values=vertex_rgb)
│   │   └── impls effective_color = those CSS colors, one per vertex
│   ├── impls trace_kwargs = the verts' x, y and z with the faces' i, j and k, at effective_opacity, smooth-shaded, hover skipped  # impls-node-one-step:skip — names trace fields
│   ├── if effective_color is a list
│   │   └── impls trace_kwargs["vertexcolor"] = effective_color
│   ├── else
│   │   └── impls trace_kwargs["color"] = effective_color
│   ├── impls scene = the go.Mesh3d over trace_kwargs
│   └── return scene
├── def _create_dash_uv_texture_map_mesh_scene(mesh: Any, mesh_color: Optional[str], effective_opacity: float, effective_side: str) -> go.Mesh3d
│   ├── # Sync-builds the Plotly Mesh3d trace from a UV-textured mesh, sampling its texture per vertex unless mesh_color overrides it.
│   ├── assert mesh is a Mesh                             # reporting its type
│   ├── assert mesh.texture is a MeshTextureUVTextureMap  # reporting its type
│   ├── assert mesh_color is None or a str                # reporting its type
│   ├── assert effective_opacity is numeric               # reporting its type
│   ├── assert effective_side is a str                    # reporting its type
│   ├── impls verts_np = mesh.verts.detach().cpu().numpy()
│   ├── impls faces_np = mesh.faces.detach().cpu().numpy()
│   ├── if mesh_color is not None
│   │   └── impls effective_color = mesh_color
│   ├── else
│   │   ├── calls _normalize_texture_map_to_uint8(texture_map=mesh.texture.uv_texture_map)  # -> texture_map
│   │   ├── impls (texture_height, texture_width) = the texture map's first two dimensions
│   │   ├── impls verts_uvs_np = mesh.texture.verts_uvs.detach().cpu().numpy()
│   │   ├── impls faces_uvs_np = mesh.texture.faces_uvs.detach().cpu().numpy()
│   │   ├── impls vertex_uvs = a (V, 2) array of zeros in the UV table's dtype
│   │   ├── impls vertex_uvs[the faces' corners] = the UV faces' own corners  # collapses the per-face UV layer onto one UV per vertex
│   │   ├── impls column = vertex_uvs[:, 0] scaled by texture_width - 1, rounded to int64 and clamped into [0, texture_width - 1]
│   │   ├── impls row = (1 - vertex_uvs[:, 1]) scaled by texture_height - 1, rounded to int64 and clamped into [0, texture_height - 1]
│   │   ├── impls sampled_rgb = texture_map[row, column]
│   │   ├── for vertex_rgb in sampled_rgb
│   │   │   └── calls _rgb_to_css_color(rgb_values=vertex_rgb)
│   │   └── impls effective_color = those CSS colors, one per vertex
│   ├── impls trace_kwargs = the verts' x, y and z with the faces' i, j and k, at effective_opacity, smooth-shaded, hover skipped  # impls-node-one-step:skip — names trace fields
│   ├── if effective_color is a list
│   │   └── impls trace_kwargs["vertexcolor"] = effective_color
│   ├── else
│   │   └── impls trace_kwargs["color"] = effective_color
│   ├── impls scene = the go.Mesh3d over trace_kwargs
│   └── return scene
├── def _normalize_rgb_tensor_to_uint8(rgb_values: torch.Tensor) -> np.ndarray
│   ├── # Normalizes one RGB tensor to the uint8 (N, 3) numpy layout.
│   └── return
├── def _normalize_texture_map_to_uint8(texture_map: torch.Tensor) -> np.ndarray
│   ├── # Normalizes one UV texture tensor to the uint8 (H, W, 3) numpy layout.
│   └── return
├── def _rgb_to_css_color(rgb_values: np.ndarray) -> str
│   ├── # Converts one RGB triplet to its CSS color string.
│   ├── impls css_color = that triplet as `rgb(r,g,b)`
│   └── return css_color
├── def _build_textured_triangle_buffers(verts: np.ndarray, faces: np.ndarray, verts_uvs: np.ndarray, faces_uvs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]
│   ├── # Builds the exploded per-corner position and UV buffers of one UV-textured mesh.
│   └── return
├── def _build_texture_data_url(texture_map: np.ndarray) -> str
│   ├── # Encodes one texture map as an inline PNG data URL.
│   ├── impls texture_image = Image.fromarray(texture_map, mode="RGB")
│   ├── impls texture_buffer = io.BytesIO()
│   ├── impls texture_image.save(texture_buffer, format="PNG")
│   ├── impls texture_base64 = that buffer, base64-encoded as ascii
│   ├── impls texture_data_url = that base64 under the PNG data-url prefix
│   └── return texture_data_url
└── def create_dash_mesh_component(scene: go.Mesh3d, controls: Any) -> dcc.Graph
    ├── # Wraps the mesh scene into a Dash graph over a one-trace figure.
    ├── assert scene is a go.Mesh3d  # reporting its type
    ├── impls component = the dcc.Graph over the figure holding that scene
    └── return component
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
├── const DEFAULT_MESH_COLOR = "#cccccc"                    # uniform fallback when the geometry has no texture and no vertex colors and the caller supplies no meshColor; lib-owned default, overridable
├── const DEFAULT_MESH_OPACITY = 1.0                        # opaque default when the caller supplies no meshOpacity; lib-owned default, overridable
├── const DEFAULT_MESH_SIDE: THREE.Side = THREE.DoubleSide  # fallback side mode keeping the mesh visible under arbitrary camera framings; lib-owned default, overridable
├── const NEUTRAL_GRAY = 0.75                               # per-channel color an OBJ vertex line carrying no color reads as
├── const _objCache = new Map<string, Promise<ParsedObj>>()          # obj url -> its in-flight or settled parse, so one url is fetched and parsed once
├── const _textureCache = new Map<string, Promise<THREE.Texture>>()  # texture url -> its in-flight or settled load, so one image is loaded once
├── interface ParsedObj
│   ├── # The logical, indexed result of parsing a Wavefront OBJ, whose `vt` count may differ from its `v` count.
│   ├── verts: Float32Array
│   ├── faces: Uint32Array
│   ├── vertexColor: Float32Array | null
│   ├── vertsUvs: Float32Array | null
│   ├── facesUvs: Uint32Array | null
│   └── mtllibName: string | null
├── interface SparseHeatmapResource
│   ├── # The wire payload of a sparse heatmap resource: a reference to the shared column geometry plus the part's non-zero delta.
│   ├── geometryUrl: string
│   ├── indices: Int32Array
│   └── values: Float32Array
├── interface MeshTextureVertexColor
│   ├── # Render mirror of the data structure's MeshTextureVertexColor: per-vertex colors aligned 1:1 with verts, 3 components for dense opaque RGB and 4 for a sparse-heatmap overlay's RGBA.
│   ├── kind: "vertex_color"
│   └── vertexColor: Float32Array
├── interface MeshTextureUVTextureMap
│   ├── # Render mirror of the data structure's MeshTextureUVTextureMap: a per-face-indexed UV texture map.
│   ├── kind: "uv_texture_map"
│   ├── uvTextureMap: THREE.Texture
│   ├── vertsUvs: Float32Array
│   └── facesUvs: Uint32Array
├── interface MeshPayload
│   ├── # The render-side mirror of the Mesh data structure: indexed geometry plus an optional polymorphic MeshTexture.
│   ├── verts: Float32Array
│   ├── faces: Uint32Array
│   └── texture: MeshTextureVertexColor | MeshTextureUVTextureMap | null
├── export function renderMeshDisplay({ displayResponse, initialCameraState = null, meshColor, meshOpacity, meshSide, }: { displayResponse: MeshDisplayResponse; initialCameraState?: CameraState | null; meshColor?: string; meshOpacity?: number; meshSide?: THREE.Side; }): LeafVNode
│   ├── # Renders a self-contained mesh display element initialized at initialCameraState.
│   ├── () => [local]
│   │   ├── # The leaf's render: mounts the mesh display and returns its container.
│   │   ├── calls createSpatialDisplayScene({ initialCameraState })                        # -> { container, scene, camera, renderer }
│   │   ├── calls createMeshObject({ displayResponse, meshColor, meshOpacity, meshSide })  # -> object
│   │   ├── impls scene.add(object)
│   │   ├── calls createTrackballCameraControls({ container, camera, renderer, initialCameraState })  # -> controls
│   │   ├── calls renderMeshScene({ scene, camera, renderer, controls })
│   │   └── return container
│   ├── impls leaf = the LeafVNode keyed by displayResponse.url or `mesh:${displayResponse.slot_id}`, with empty props and that render  # impls-node-one-step:skip — one constructor's fields
│   └── return leaf
├── export function createMeshObject({ displayResponse, meshColor, meshOpacity, meshSide, }: { displayResponse: MeshDisplayResponse; meshColor?: string; meshOpacity?: number; meshSide?: THREE.Side; }): THREE.Object3D
│   ├── # Part-B: returns a THREE.Group for the mesh, filled with the THREE.Mesh once the async payload load resolves.
│   ├── impls group = new THREE.Group()
│   ├── calls loadMeshPayload({ displayResponse })
│   ├── (payload) => [local]
│   │   ├── # On resolve: builds the mesh from the loaded payload into the already-returned group.
│   │   ├── calls createThreeMesh({ payload, displayResponse, meshColor, meshOpacity, meshSide })
│   │   └── impls group.add(that mesh)
│   ├── (error) => [local]
│   │   ├── # On rejection: throws a new Error carrying the underlying message.
│   │   ├── if error is an Error
│   │   │   └── impls error.message
│   │   ├── else
│   │   │   └── impls String(error)
│   │   ├── impls message = that text
│   │   └── throw unable to load mesh: ${message}
│   ├── impls the payload load, chained through that resolve step and that rejection step  # impls-node-one-step:skip — both steps have their own nodes
│   └── return group
├── export async function loadMeshPayload({ displayResponse, }: { displayResponse: MeshDisplayResponse; }): Promise<MeshPayload>
│   ├── # Async-loads the mesh payload served at displayResponse.url, resolving a sparse-heatmap delta against its referenced geometry.
│   ├── if displayResponse.url is null
│   │   └── throw mesh display response url is null
│   ├── if displayResponse.display_kind is "sparse_heatmap_mesh"
│   │   ├── calls _fetchSparseHeatmapResource(displayResponse.url)  # -> sparse
│   │   ├── calls _fetchObj(sparse.geometryUrl)                     # -> parsed
│   │   ├── calls _resolveSparseHeatmapPayload({ parsed, sparse })
│   │   └── return
│   ├── calls _fetchObj(displayResponse.url)                                    # -> parsed
│   ├── calls _resolveMeshTexture({ parsed, primaryUrl: displayResponse.url })  # -> texture
│   ├── impls payload = { verts: parsed.verts, faces: parsed.faces, texture }
│   └── return payload
├── export function createThreeMesh({ payload, meshColor, meshOpacity, meshSide, }: { payload: MeshPayload; displayResponse: MeshDisplayResponse; meshColor?: string; meshOpacity?: number; meshSide?: THREE.Side; }): THREE.Mesh
│   ├── # Sync-builds the non-indexed corner-domain geometry, its material, and the THREE.Mesh from a loaded payload.
│   ├── impls cornerCount = payload.faces.length
│   ├── impls geometry = new THREE.BufferGeometry()
│   ├── impls positions = new Float32Array(cornerCount * 3)
│   ├── for corner over every corner of cornerCount
│   │   ├── impls vertexIndex = payload.faces[corner]
│   │   ├── impls positions[corner * 3] = payload.verts[vertexIndex * 3]
│   │   ├── impls positions[corner * 3 + 1] = payload.verts[vertexIndex * 3 + 1]
│   │   └── impls positions[corner * 3 + 2] = payload.verts[vertexIndex * 3 + 2]
│   ├── impls geometry.setAttribute("position", the 3-component buffer over positions)
│   ├── impls geometry.userData.cornerVertexIndices = payload.faces  # this geometry's corner → vertex map
│   ├── impls effectiveOpacity = meshOpacity ?? DEFAULT_MESH_OPACITY
│   ├── impls effectiveSide = meshSide ?? DEFAULT_MESH_SIDE
│   ├── impls let useTexture: boolean
│   ├── impls let useVertexColors: boolean
│   ├── impls let rgbaVertexColors: boolean
│   ├── impls let textureMap: THREE.Texture | undefined
│   ├── impls let effectiveColor: string | undefined
│   ├── if meshColor is supplied
│   │   ├── impls useTexture = false
│   │   ├── impls useVertexColors = false
│   │   ├── impls rgbaVertexColors = false
│   │   ├── impls textureMap = undefined
│   │   └── impls effectiveColor = meshColor
│   ├── else if payload.texture is a uv_texture_map
│   │   ├── impls uvs = new Float32Array(cornerCount * 2)
│   │   ├── for corner over every corner of cornerCount
│   │   │   ├── impls uvIndex = payload.texture.facesUvs[corner]
│   │   │   ├── impls uvs[corner * 2] = payload.texture.vertsUvs[uvIndex * 2]
│   │   │   └── impls uvs[corner * 2 + 1] = payload.texture.vertsUvs[uvIndex * 2 + 1]
│   │   ├── impls geometry.setAttribute("uv", the 2-component buffer over uvs)
│   │   ├── impls useTexture = true
│   │   ├── impls useVertexColors = false
│   │   ├── impls rgbaVertexColors = false
│   │   ├── impls textureMap = payload.texture.uvTextureMap
│   │   └── impls effectiveColor = undefined
│   ├── else if payload.texture is a vertex_color
│   │   ├── impls vertexCount = payload.verts.length / 3
│   │   ├── impls components = payload.texture.vertexColor.length / max(1, vertexCount)
│   │   ├── impls colors = new Float32Array(cornerCount * components)
│   │   ├── for corner over every corner of cornerCount
│   │   │   ├── impls vertexIndex = payload.faces[corner]
│   │   │   └── for component over every component of components
│   │   │       └── impls colors[corner * components + component] = payload.texture.vertexColor[vertexIndex * components + component]
│   │   ├── impls geometry.setAttribute("color", the components-wide buffer over colors)
│   │   ├── impls useTexture = false
│   │   ├── impls useVertexColors = true
│   │   ├── impls rgbaVertexColors = whether components is 4
│   │   ├── impls textureMap = undefined
│   │   └── impls effectiveColor = undefined
│   ├── else
│   │   ├── impls useTexture = false
│   │   ├── impls useVertexColors = false
│   │   ├── impls rgbaVertexColors = false
│   │   ├── impls textureMap = undefined
│   │   └── impls effectiveColor = DEFAULT_MESH_COLOR
│   ├── impls geometry.computeVertexNormals()
│   ├── impls geometry.computeBoundingBox()
│   ├── impls geometry.computeBoundingSphere()
│   ├── if useTexture
│   │   └── impls { map: textureMap }
│   ├── else
│   │   └── impls {}
│   ├── if effectiveColor is supplied
│   │   └── impls { color: effectiveColor }
│   ├── else
│   │   └── impls {}
│   ├── impls material = new THREE.MeshBasicMaterial(vertexColors useVertexColors, side effectiveSide, opacity effectiveOpacity, transparent when effectiveOpacity < 1 or the vertex colors carry alpha, spread with those map and color entries)  # impls-node-one-step:skip — one constructor's arguments
│   ├── impls mesh = new THREE.Mesh(geometry, material)
│   └── return mesh
├── export function renderMeshScene({ scene, camera, renderer, controls, }: { scene: THREE.Scene; camera: THREE.PerspectiveCamera; renderer: THREE.WebGLRenderer; controls: ReturnType<typeof createTrackballCameraControls>; }): void
│   ├── # Mounts the render loop for the composed mesh scene.
│   └── calls startThreeSceneRenderLoop({ scene, camera, renderer, controls })
├── async function _resolveMeshTexture({ parsed, primaryUrl, }: { parsed: ParsedObj; primaryUrl: string; }): Promise<MeshTextureVertexColor | MeshTextureUVTextureMap | null>
│   ├── # Resolves the parsed OBJ into a UV texture map, else a per-vertex color texture, else no texture.
│   ├── if parsed carries both UV layers and an mtllib name
│   │   ├── calls _siblingUrl(primaryUrl, parsed.mtllibName)
│   │   ├── calls _fetchMtlTextureName(that mtl url)  # -> textureName
│   │   ├── if textureName is null
│   │   │   └── throw mesh OBJ declares UVs but its MTL has no map_Kd: ${primaryUrl}
│   │   ├── calls _siblingUrl(primaryUrl, textureName)
│   │   ├── calls _fetchTexture(that texture url)  # -> uvTextureMap
│   │   ├── impls texture = { kind: "uv_texture_map", uvTextureMap, vertsUvs: parsed.vertsUvs, facesUvs: parsed.facesUvs }
│   │   └── return texture
│   ├── if parsed.vertexColor is not null
│   │   ├── impls texture = { kind: "vertex_color", vertexColor: parsed.vertexColor }
│   │   └── return texture
│   └── return null
├── function _fetchObj(url: string): Promise<ParsedObj>
│   ├── # Fetches and parses a Wavefront OBJ url once; subsequent callers share the promise.
│   ├── impls cached = _objCache.get(url)
│   ├── if cached is not undefined
│   │   └── return cached
│   ├── async (response) => [local]
│   │   ├── # On response: parses the OBJ text.
│   │   ├── if the response is not ok
│   │   │   └── throw GET ${url} failed: ${response.status}
│   │   ├── calls _parseObj(the awaited response text)
│   │   └── return
│   ├── impls promise = the fetch of url, chained through that step
│   ├── impls _objCache.set(url, promise)
│   └── return promise
├── function _parseObj(text: string): ParsedObj
│   ├── # Parses a Wavefront OBJ into logical indexed geometry: verts, triangulated faces, optional per-vertex colors, and an optional per-face UV indexing layer.
│   ├── impls vPositions = []
│   ├── impls vColors = []
│   ├── impls vtCoords = []
│   ├── impls sawVertexColors = false
│   ├── impls mtllibName = null
│   ├── impls faceVertexTokens = []
│   ├── impls faceUvTokens = []
│   ├── impls sawAnyUv = false
│   ├── for each raw line of text split on newlines
│   │   ├── if raw is empty
│   │   │   └── continue
│   │   ├── impls c0 = raw.charCodeAt(0)
│   │   ├── if raw is a `v ` line
│   │   │   ├── impls parts = the trimmed raw split on whitespace
│   │   │   ├── impls vPositions.push(parts[1], parts[2], parts[3] as floats)
│   │   │   ├── if parts holds 7 entries or more
│   │   │   │   ├── impls sawVertexColors = true
│   │   │   │   └── impls vColors.push(parts[4], parts[5], parts[6] as floats)
│   │   │   └── else
│   │   │       └── impls vColors.push(NEUTRAL_GRAY, NEUTRAL_GRAY, NEUTRAL_GRAY)
│   │   ├── else if raw is a `vt` line
│   │   │   ├── impls parts = the trimmed raw split on whitespace
│   │   │   └── impls vtCoords.push(parts[1], parts[2] as floats)
│   │   ├── else if raw is an `f ` line
│   │   │   ├── impls parts = the trimmed raw split on whitespace
│   │   │   ├── impls corners = []
│   │   │   ├── for p over every token of parts past the first
│   │   │   │   ├── calls _parseFaceCorner(parts[p])
│   │   │   │   └── impls corners.push(that corner)
│   │   │   └── for j over every fan triangle, from 1 to corners.length - 2
│   │   │       ├── impls fan = [corners[0], corners[j], corners[j + 1]]
│   │   │       └── for each corner in fan
│   │   │           ├── impls faceVertexTokens.push(corner.v)
│   │   │           ├── impls faceUvTokens.push(corner.vt)
│   │   │           └── if corner.vt >= 0
│   │   │               └── impls sawAnyUv = true
│   │   └── else if raw starts with "mtllib"
│   │       ├── impls parts = the trimmed raw split on whitespace
│   │       └── if parts holds 2 entries or more
│   │           └── impls mtllibName = the parts past the first, joined on a space
│   ├── impls geometryVertexCount = vPositions.length / 3
│   ├── impls cornerCount = faceVertexTokens.length
│   ├── impls useUvs = sawAnyUv and vtCoords is non-empty  # impls-node-one-step:skip — one boolean expression
│   ├── impls verts = new Float32Array(vPositions)
│   ├── impls faces = new Uint32Array(cornerCount)
│   ├── if sawVertexColors
│   │   └── impls new Float32Array(vColors)
│   ├── else
│   │   └── impls null
│   ├── impls vertexColor = that value
│   ├── if useUvs
│   │   └── impls new Uint32Array(cornerCount)
│   ├── else
│   │   └── impls null
│   ├── impls facesUvs = that value
│   ├── if useUvs
│   │   └── impls new Float32Array(vtCoords)
│   ├── else
│   │   └── impls null
│   ├── impls vertsUvs = that value
│   ├── for corner over every corner of cornerCount
│   │   ├── impls vIndex = faceVertexTokens[corner]
│   │   ├── if vIndex is outside [0, geometryVertexCount)
│   │   │   └── throw OBJ face references out-of-range vertex index: ${vIndex}
│   │   ├── impls faces[corner] = vIndex
│   │   └── if facesUvs is not null
│   │       ├── impls vtIndex = faceUvTokens[corner]
│   │       ├── if vtIndex is below 0 or its second coordinate is past the end of vtCoords
│   │       │   └── throw OBJ face corner is missing a valid UV index: ${vtIndex}
│   │       └── impls facesUvs[corner] = vtIndex
│   ├── impls parsed = { verts, faces, vertexColor, vertsUvs, facesUvs, mtllibName }
│   └── return parsed
├── function _parseFaceCorner(token: string): { v: number; vt: number }
│   ├── # Parses one OBJ face token into its 0-based vertex and UV indices, -1 where the token carries no UV.
│   ├── impls fields = token split on "/"
│   ├── impls v = fields[0] read as an integer, less 1
│   ├── if fields holds a second, non-empty entry
│   │   └── impls that entry read as an integer, less 1
│   ├── else
│   │   └── impls -1
│   ├── impls vt = that value
│   ├── impls corner = { v, vt }
│   └── return corner
├── async function _fetchMtlTextureName(mtlUrl: string): Promise<string | null>
│   ├── # Fetches a `.mtl` sibling and reads its map_Kd texture-image filename.
│   ├── impls response = await fetch(mtlUrl)
│   ├── if the response is not ok
│   │   └── throw GET ${mtlUrl} failed: ${response.status}
│   ├── impls text = await response.text()
│   ├── for each raw line of text split on newlines
│   │   ├── impls line = the trimmed raw
│   │   └── if line starts with "map_Kd"
│   │       ├── impls parts = line split on whitespace
│   │       └── if parts holds 2 entries or more
│   │           ├── impls textureName = the parts past the first, joined on a space
│   │           └── return textureName
│   └── return null
├── function _fetchTexture(textureUrl: string): Promise<THREE.Texture>
│   ├── # Loads a texture image url once into a THREE.Texture; subsequent callers share the promise.
│   ├── impls cached = _textureCache.get(textureUrl)
│   ├── if cached is not undefined
│   │   └── return cached
│   ├── impls loader = new THREE.TextureLoader()
│   ├── (resolve, reject) => [local]
│   │   ├── # The promise executor: starts the load and settles on its outcome.
│   │   ├── (texture: THREE.Texture) => [local]
│   │   │   ├── # On load: stamps the color space and flip, then resolves.
│   │   │   ├── impls texture.colorSpace = THREE.SRGBColorSpace
│   │   │   ├── impls texture.flipY = true
│   │   │   ├── impls texture.needsUpdate = true
│   │   │   └── impls resolve(texture)
│   │   ├── () => [local]
│   │   │   ├── # On failure: rejects with an Error naming the unloadable image url.
│   │   │   └── impls reject(new Error(`unable to load texture image: ${textureUrl}`))
│   │   └── impls loader.load(textureUrl, that load step, undefined, that failure step)
│   ├── impls promise = the promise over that executor
│   ├── impls _textureCache.set(textureUrl, promise)
│   └── return promise
├── function _siblingUrl(primaryUrl: string, siblingName: string): string
│   ├── # Resolves a sibling resource url relative to the primary url.
│   ├── impls slash = primaryUrl.lastIndexOf("/")
│   ├── if slash < 0
│   │   └── return siblingName
│   ├── impls siblingUrl = primaryUrl through that slash, followed by siblingName
│   └── return siblingUrl
├── async function _fetchSparseHeatmapResource(url: string): Promise<SparseHeatmapResource>
│   ├── # Fetches and decodes a sparse heatmap wire resource: its geometry reference and its delta.
│   ├── impls response = await fetch(url)
│   ├── if the response is not ok
│   │   └── throw GET ${url} failed: ${response.status}
│   ├── impls raw = the awaited response body parsed as JSON, read as its geometry_url, indices and values fields  # impls-node-one-step:skip — names three fields
│   ├── if raw.geometry_url is not a non-empty string
│   │   └── throw sparse heatmap resource is missing geometry_url: ${url}
│   ├── if either of raw.indices and raw.values is not an array
│   │   └── throw sparse heatmap resource is missing indices/values arrays: ${url}
│   ├── impls resource = { geometryUrl: raw.geometry_url, indices as an Int32Array, values as a Float32Array }
│   └── return resource
├── function _resolveSparseHeatmapPayload({ parsed, sparse, }: { parsed: ParsedObj; sparse: SparseHeatmapResource; }): MeshPayload
│   ├── # Resolves a sparse heatmap delta against its geometry into a per-vertex RGBA overlay, alpha 0 outside the delta so the base layer shows through.
│   ├── impls vertexCount = parsed.verts.length / 3
│   ├── impls vertexColor = new Float32Array(vertexCount * 4)
│   ├── calls _mapScalarsToRgb(sparse.values)  # -> rgb
│   ├── for i over every entry of sparse.indices
│   │   ├── impls vertexIndex = sparse.indices[i]
│   │   ├── if vertexIndex is outside [0, vertexCount)
│   │   │   └── throw sparse heatmap references out-of-range vertex index: ${vertexIndex} (vertexCount=${vertexCount})
│   │   ├── impls vertexColor[vertexIndex * 4] = rgb[i * 3] / 255
│   │   ├── impls vertexColor[vertexIndex * 4 + 1] = rgb[i * 3 + 1] / 255
│   │   ├── impls vertexColor[vertexIndex * 4 + 2] = rgb[i * 3 + 2] / 255
│   │   └── impls vertexColor[vertexIndex * 4 + 3] = 1.0
│   ├── impls payload = { verts: parsed.verts, faces: parsed.faces, texture: { kind: "vertex_color", vertexColor } }
│   └── return payload
├── const _HEATMAP_PALETTE_STOPS: ReadonlyArray<number> = [0.0, 0.25, 0.5, 0.75, 1.0]  # the normalized stops the palette interpolates between
├── const _HEATMAP_PALETTE_COLORS: ReadonlyArray<readonly [number, number, number]> = [[0, 0, 255], [0, 255, 255], [0, 255, 0], [255, 255, 0], [255, 0, 0]]  # the RGB color at each stop, blue through red
└── function _mapScalarsToRgb(values: Float32Array): Uint8Array
    ├── # Maps non-negative scalars to RGB through the fixed continuous heatmap palette, normalized by the largest value.
    ├── impls maxValue = 0.0
    ├── for i over every entry of values
    │   └── if values[i] > maxValue
    │       └── impls maxValue = values[i]
    ├── impls denom = max(maxValue, 1e-12)
    ├── impls rgb = new Uint8Array(values.length * 3)
    ├── for i over every entry of values
    │   ├── impls normalized = values[i] / denom, clamped into [0, 1]
    │   ├── impls segment = 0
    │   ├── while segment is below the last palette segment and normalized reaches the next stop
    │   │   └── impls segment += 1
    │   ├── impls left = _HEATMAP_PALETTE_STOPS[segment]
    │   ├── impls right = _HEATMAP_PALETTE_STOPS[segment + 1]
    │   ├── impls extent = max(right - left, 1e-12)
    │   ├── impls fraction = (normalized - left) / extent, clamped into [0, 1]
    │   ├── impls c0 = _HEATMAP_PALETTE_COLORS[segment]
    │   ├── impls c1 = _HEATMAP_PALETTE_COLORS[segment + 1]
    │   ├── impls rgb[i * 3] = round(c0[0] + (c1[0] - c0[0]) * fraction)
    │   ├── impls rgb[i * 3 + 1] = round(c0[1] + (c1[1] - c0[1]) * fraction)
    │   └── impls rgb[i * 3 + 2] = round(c0[2] + (c1[2] - c0[2]) * fraction)
    └── return rgb
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
