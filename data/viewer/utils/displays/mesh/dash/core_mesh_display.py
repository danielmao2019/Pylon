"""Mesh display utilities for triangle-mesh visualization."""

import base64
import io
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import plotly.graph_objects as go
import torch
from dash import dcc, html
from PIL import Image

from data.structures.three_d.mesh.mesh import Mesh
from data.structures.three_d.mesh.texture.mesh_texture_uv_texture_map import (
    MeshTextureUVTextureMap,
)
from data.structures.three_d.mesh.texture.mesh_texture_vertex_color import (
    MeshTextureVertexColor,
)
from data.viewer.utils.controls.camera.camera_controls.dash.trackball_camera_controls import (
    create_dash_trackball_camera_controls,
)
from data.viewer.utils.controls.camera.camera_sync.threejs import (
    build_threejs_camera_sync_script,
)

# Uniform fallback color used when geometry has no texture AND has no per-vertex
# colors AND the caller does not supply `mesh_color`. Lib-owned default,
# overridable by the caller.
DEFAULT_MESH_COLOR = "#cccccc"
# Opaque default applied when the caller does not supply `mesh_opacity`.
# Lib-owned default, overridable by the caller.
DEFAULT_MESH_OPACITY = 1.0
# Fallback side mode for visibility under arbitrary camera framings when the
# caller does not supply `mesh_side`. Lib-owned default, overridable.
DEFAULT_MESH_SIDE = "double"

MODULE_DIR = Path(__file__).resolve().parent
TEXTURED_MESH_VIEWER_SCRIPT_PATH = MODULE_DIR / "mesh_display_textured_viewer.js"
PLOTLY_MESH_DEFAULT_CAMERA_EYE_Z = 2.4
THREEJS_MESH_DEFAULT_CAMERA_EYE_Z = PLOTLY_MESH_DEFAULT_CAMERA_EYE_Z
PLOTLY_MESH_CAMERA_UIREVISION = "mesh-display-camera"
MESH_VIEW_BOUNDS_KEYS = {
    "center",
    "camera_coordinate_scale",
    "half_span",
    "max_span",
    "axis_ranges",
}
TEXTURED_MESH_IFRAME_STYLE: Dict[str, str] = {
    "width": "100%",
    "height": "100%",
    "minHeight": "0",
    "border": "1px solid #dce5f0",
    "borderRadius": "12px",
    "overflow": "hidden",
    "backgroundColor": "#f7fafc",
}


def create_mesh_display(
    mesh: Mesh,
    title: str,
    component_id: Optional[str] = None,
    camera_sync_group: Optional[str] = None,
) -> Union[go.Figure, html.Iframe]:
    """Create one display component from a generic mesh container.

    Args:
        mesh: Mesh with either per-vertex RGB colors or a UV texture map.
        title: Display title.
        component_id: Optional Dash component id for non-Plotly mesh displays.
        camera_sync_group: Optional browser-side camera-sync group id.

    Returns:
        Plotly figure for vertex-color meshes or a Three.js iframe for UV-texture meshes.
    """

    def _validate_inputs() -> None:
        assert isinstance(mesh, Mesh), (
            "Expected `mesh` to be a `Mesh` instance. " f"{type(mesh)=}"
        )
        assert isinstance(
            mesh.texture, (MeshTextureVertexColor, MeshTextureUVTextureMap)
        ), (
            "Expected `mesh` to carry a `MeshTextureVertexColor` or "
            "`MeshTextureUVTextureMap` texture. "
            f"{type(mesh.texture)=}"
        )

        assert isinstance(title, str), (
            "Expected `title` to be a string. " f"{type(title)=}"
        )
        assert title.strip() != "", (
            "Expected `title` to be non-empty after trimming. " f"{title=}"
        )

        assert component_id is None or isinstance(component_id, str), (
            "Expected `component_id` to be `None` or a string. "
            f"{type(component_id)=}"
        )
        if component_id is not None:
            assert component_id.strip() != "", (
                "Expected `component_id` to be non-empty after trimming. "
                f"{component_id=}"
            )

        assert camera_sync_group is None or isinstance(camera_sync_group, str), (
            "Expected `camera_sync_group` to be `None` or a string. "
            f"{type(camera_sync_group)=}"
        )
        if camera_sync_group is not None:
            assert camera_sync_group.strip() != "", (
                "Expected `camera_sync_group` to be non-empty after trimming. "
                f"{camera_sync_group=}"
            )

    _validate_inputs()

    def _normalize_inputs() -> Tuple[str, str, Optional[str]]:
        normalized_title = title.strip()

        normalized_component_id = _normalize_component_id(
            component_id=component_id,
            title=normalized_title,
        )

        normalized_camera_sync_group = None
        if camera_sync_group is not None:
            normalized_camera_sync_group = camera_sync_group.strip()

        return (
            normalized_title,
            normalized_component_id,
            normalized_camera_sync_group,
        )

    normalized_title, normalized_component_id, normalized_camera_sync_group = (
        _normalize_inputs()
    )

    if isinstance(mesh.texture, MeshTextureVertexColor):
        return _create_vertex_color_mesh_display(
            mesh=mesh,
            title=normalized_title,
        )

    return _create_uv_texture_mesh_display(
        mesh=mesh,
        title=normalized_title,
        component_id=normalized_component_id,
        camera_sync_group=normalized_camera_sync_group,
    )


def _create_vertex_color_mesh_display(
    mesh: Mesh,
    title: str,
) -> go.Figure:
    """Create one Plotly mesh figure from per-vertex colors.

    Args:
        mesh: Mesh carrying a `MeshTextureVertexColor` texture.
        title: Figure title.

    Returns:
        Plotly figure with one vertex-colored mesh trace.
    """

    def _validate_inputs() -> None:
        assert isinstance(mesh, Mesh), (
            "Expected `_create_vertex_color_mesh_display` to receive a `Mesh`. "
            f"{type(mesh)=}"
        )
        assert isinstance(mesh.texture, MeshTextureVertexColor), (
            "Expected `_create_vertex_color_mesh_display` to receive a "
            "`MeshTextureVertexColor` texture. "
            f"{type(mesh.texture)=}"
        )

        assert isinstance(title, str), (
            "Expected `title` to be a string. " f"{type(title)=}"
        )
        assert title != "", "Expected `title` to be non-empty. " f"{title=}"

    _validate_inputs()

    def _normalize_inputs() -> (
        Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]
    ):
        normalized_verts = mesh.verts.detach().cpu().numpy()
        normalized_faces = mesh.faces.detach().cpu().numpy()
        normalized_vertex_colors = _normalize_rgb_tensor_to_uint8(
            rgb_values=mesh.texture.vertex_color
        )
        normalized_mesh_view_bounds = build_mesh_view_bounds(
            verts=mesh.verts,
        )
        return (
            normalized_verts,
            normalized_faces,
            normalized_vertex_colors,
            normalized_mesh_view_bounds,
        )

    verts_np, faces_np, vertex_colors, mesh_view_bounds = _normalize_inputs()

    figure = go.Figure(
        data=[
            go.Mesh3d(
                x=verts_np[:, 0],
                y=verts_np[:, 1],
                z=verts_np[:, 2],
                i=faces_np[:, 0],
                j=faces_np[:, 1],
                k=faces_np[:, 2],
                vertexcolor=[
                    _rgb_to_css_color(rgb_values=vertex_rgb)
                    for vertex_rgb in vertex_colors
                ],
                flatshading=False,
                lighting={
                    "ambient": 0.9,
                    "diffuse": 0.5,
                    "specular": 0.05,
                    "roughness": 0.8,
                    "fresnel": 0.02,
                },
                lightposition={"x": 120.0, "y": 80.0, "z": 240.0},
                hoverinfo="skip",
                name=title,
            )
        ]
    )
    _apply_mesh_layout(
        figure=figure,
        title=title,
        mesh_view_bounds=mesh_view_bounds,
    )
    return figure


def _create_uv_texture_mesh_display(
    mesh: Mesh,
    title: str,
    component_id: str,
    camera_sync_group: Optional[str],
) -> html.Iframe:
    """Create one Three.js iframe from a UV texture map.

    Args:
        mesh: Mesh carrying a `MeshTextureUVTextureMap` texture.
        title: Display title.
        component_id: Dash component id for the iframe.
        camera_sync_group: Optional browser-side camera-sync group id.

    Returns:
        Dash iframe with one textured-mesh viewer.
    """

    def _validate_inputs() -> None:
        assert isinstance(mesh, Mesh), (
            "Expected `_create_uv_texture_mesh_display` to receive a `Mesh`. "
            f"{type(mesh)=}"
        )
        assert isinstance(mesh.texture, MeshTextureUVTextureMap), (
            "Expected `_create_uv_texture_mesh_display` to receive a "
            "`MeshTextureUVTextureMap` texture. "
            f"{type(mesh.texture)=}"
        )

        assert isinstance(title, str), (
            "Expected `title` to be a string. " f"{type(title)=}"
        )
        assert title != "", "Expected `title` to be non-empty. " f"{title=}"

        assert isinstance(component_id, str), (
            "Expected `component_id` to be a string. " f"{type(component_id)=}"
        )
        assert component_id != "", (
            "Expected `component_id` to be non-empty. " f"{component_id=}"
        )

        assert camera_sync_group is None or isinstance(camera_sync_group, str), (
            "Expected `camera_sync_group` to be `None` or a string. "
            f"{type(camera_sync_group)=}"
        )

    _validate_inputs()

    def _normalize_inputs() -> Tuple[List[float], List[float], str, Dict[str, object]]:
        display_mesh = mesh.to(verts_uvs_convention="obj")
        normalized_verts = display_mesh.verts.detach().cpu().numpy()
        normalized_faces = display_mesh.faces.detach().cpu().numpy()
        normalized_verts_uvs = display_mesh.texture.verts_uvs.detach().cpu().numpy()
        normalized_faces_uvs = display_mesh.texture.faces_uvs.detach().cpu().numpy()
        normalized_mesh_view_bounds = build_mesh_view_bounds(
            verts=display_mesh.verts,
        )
        normalized_triangle_positions, normalized_triangle_uvs = (
            _build_textured_triangle_buffers(
                verts=normalized_verts,
                faces=normalized_faces,
                verts_uvs=normalized_verts_uvs,
                faces_uvs=normalized_faces_uvs,
            )
        )
        normalized_texture_map = _normalize_texture_map_to_uint8(
            texture_map=display_mesh.texture.uv_texture_map,
        )
        normalized_texture_data_url = _build_texture_data_url(
            texture_map=normalized_texture_map,
        )
        return (
            normalized_triangle_positions.tolist(),
            normalized_triangle_uvs.tolist(),
            normalized_texture_data_url,
            normalized_mesh_view_bounds,
        )

    (
        triangle_position_values,
        triangle_uv_values,
        texture_data_url,
        mesh_view_bounds,
    ) = _normalize_inputs()

    iframe_html = _build_textured_mesh_html(
        title=title,
        position_values=triangle_position_values,
        uv_values=triangle_uv_values,
        texture_data_url=texture_data_url,
        mesh_view_bounds=mesh_view_bounds,
        viewer_id=component_id,
        camera_sync_group=camera_sync_group,
    )
    iframe_attributes: Dict[str, str] = {}
    if camera_sync_group is not None:
        iframe_attributes = {
            "data-camera-sync-group": camera_sync_group,
            "data-camera-sync-viewer-id": component_id,
        }
    return html.Iframe(
        id=component_id,
        srcDoc=iframe_html,
        style=TEXTURED_MESH_IFRAME_STYLE,
        **iframe_attributes,
    )


def _normalize_component_id(
    component_id: Optional[str],
    title: str,
) -> str:
    """Normalize one optional component id.

    Args:
        component_id: Optional caller-provided component id.
        title: Display title used to derive a fallback id.

    Returns:
        Normalized component id string.
    """

    def _validate_inputs() -> None:
        assert component_id is None or isinstance(component_id, str), (
            "Expected `component_id` to be `None` or a string. "
            f"{type(component_id)=}"
        )
        if component_id is not None:
            assert component_id.strip() != "", (
                "Expected `component_id` to be non-empty after trimming. "
                f"{component_id=}"
            )

        assert isinstance(title, str), (
            "Expected `title` to be a string. " f"{type(title)=}"
        )
        assert title != "", "Expected `title` to be non-empty. " f"{title=}"

    _validate_inputs()

    def _normalize_inputs() -> str:
        if component_id is not None:
            return component_id.strip()

        normalized_characters = [
            character.lower() if character.isalnum() else "-" for character in title
        ]
        normalized_component_id = "".join(normalized_characters).strip("-")
        while "--" in normalized_component_id:
            normalized_component_id = normalized_component_id.replace("--", "-")
        assert normalized_component_id != "", (
            "Expected the fallback `component_id` derived from `title` to be non-empty. "
            f"{title=}"
        )
        return f"mesh-display-{normalized_component_id}"

    return _normalize_inputs()


def build_mesh_view_bounds(
    verts: torch.Tensor,
) -> Dict[str, object]:
    """Build one renderer framing summary from raw mesh verts.

    Args:
        verts: Mesh vertex tensor with shape `[V, 3]`.

    Returns:
        Dict containing the raw bounds center, renderer camera-coordinate scale,
        max span, and raw per-axis ranges.
    """

    def _validate_inputs() -> None:
        assert isinstance(verts, torch.Tensor), (
            "Expected `verts` to be a tensor. " f"{type(verts)=}"
        )
        assert verts.ndim == 2 and verts.shape[1] == 3, (
            "Expected `verts` shape `[V, 3]`. " f"{verts.shape=}"
        )

    _validate_inputs()

    def _normalize_inputs() -> Dict[str, object]:
        min_corner = verts.min(dim=0, keepdim=True).values
        max_corner = verts.max(dim=0, keepdim=True).values
        bounds_center = (min_corner + max_corner) / 2.0
        bounds_extent = max_corner - min_corner
        max_extent = float(bounds_extent.max().item())
        assert max_extent > 0.0, (
            "Expected mesh bounds to have a positive extent before building view bounds. "
            f"{bounds_extent=}"
        )
        camera_coordinate_scale = _compute_camera_coordinate_scale(
            bounds_extent=bounds_extent,
        )
        half_span = max_extent / 2.0
        center_values = bounds_center.reshape(-1).detach().cpu().tolist()
        min_corner_values = min_corner.reshape(-1).detach().cpu().tolist()
        max_corner_values = max_corner.reshape(-1).detach().cpu().tolist()
        return {
            "center": {
                "x": float(center_values[0]),
                "y": float(center_values[1]),
                "z": float(center_values[2]),
            },
            "camera_coordinate_scale": float(camera_coordinate_scale),
            "half_span": float(half_span),
            "max_span": float(max_extent),
            "axis_ranges": {
                "x": [float(min_corner_values[0]), float(max_corner_values[0])],
                "y": [float(min_corner_values[1]), float(max_corner_values[1])],
                "z": [float(min_corner_values[2]), float(max_corner_values[2])],
            },
        }

    return _normalize_inputs()


def _compute_camera_coordinate_scale(
    bounds_extent: torch.Tensor,
) -> float:
    """Compute one shared camera-coordinate scale from mesh axis extents.

    Args:
        bounds_extent: Non-negative mesh extent tensor with shape `[1, 3]` or `[3]`.

    Returns:
        Positive scale factor that maps raw mesh coordinates into Plotly scene units.
    """

    def _validate_inputs() -> None:
        assert isinstance(bounds_extent, torch.Tensor), (
            "Expected `bounds_extent` to be a tensor when computing the camera "
            f"coordinate scale. {type(bounds_extent)=}"
        )
        assert bounds_extent.numel() == 3, (
            "Expected `bounds_extent` to contain exactly three axis extents when "
            f"computing the camera coordinate scale. {bounds_extent.shape=}"
        )

    _validate_inputs()

    flattened_extent = bounds_extent.reshape(-1).detach().cpu().to(torch.float64)
    positive_extent = flattened_extent[flattened_extent > 0.0]
    assert positive_extent.numel() > 0, (
        "Expected at least one positive axis extent when computing the camera "
        f"coordinate scale. {flattened_extent=}"
    )
    return float(torch.exp(torch.log(positive_extent).mean()).item())


def validate_mesh_view_bounds(
    mesh_view_bounds: Dict[str, object],
) -> None:
    """Validate one shared mesh-view-bounds payload.

    Args:
        mesh_view_bounds: Renderer framing summary derived from mesh bounds.

    Returns:
        None.
    """

    assert isinstance(mesh_view_bounds, dict), (
        "Expected `mesh_view_bounds` to be a dict. " f"{type(mesh_view_bounds)=}"
    )
    assert set(mesh_view_bounds.keys()) == MESH_VIEW_BOUNDS_KEYS, (
        "Expected `mesh_view_bounds` to match the shared mesh-view-bounds "
        f"contract. {mesh_view_bounds=}"
    )


def _normalize_rgb_tensor_to_uint8(
    rgb_values: torch.Tensor,
) -> np.ndarray:
    """Normalize one RGB tensor to uint8 numpy layout.

    Args:
        rgb_values: RGB tensor with shape `[N, 3]` in uint8 `[0, 255]` or float32 `[0, 1]`.

    Returns:
        RGB numpy array with dtype `uint8` and shape `(N, 3)`.
    """

    def _validate_inputs() -> None:
        assert isinstance(rgb_values, torch.Tensor), (
            "Expected `rgb_values` to be a tensor. " f"{type(rgb_values)=}"
        )
        assert rgb_values.ndim == 2 and rgb_values.shape[1] == 3, (
            "Expected `rgb_values` shape `[N, 3]`. " f"{rgb_values.shape=}"
        )
        assert rgb_values.dtype in {torch.uint8, torch.float32}, (
            "Expected RGB tensors to be either uint8 or float32. "
            f"{rgb_values.dtype=}"
        )

    _validate_inputs()

    def _normalize_inputs() -> np.ndarray:
        if rgb_values.dtype == torch.uint8:
            return rgb_values.detach().cpu().numpy()

        assert rgb_values.dtype == torch.float32, (
            "Expected float RGB normalization to receive float32 values. "
            f"{rgb_values.dtype=}"
        )
        return (
            rgb_values.detach().cpu().clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8)
        ).numpy()

    return _normalize_inputs()


def _normalize_texture_map_to_uint8(
    texture_map: torch.Tensor,
) -> np.ndarray:
    """Normalize one UV texture tensor to uint8 HWC numpy layout.

    Args:
        texture_map: UV texture tensor with shape `[H, W, 3]`.

    Returns:
        Texture map as `(H, W, 3)` uint8 numpy array.
    """

    def _validate_inputs() -> None:
        assert isinstance(texture_map, torch.Tensor), (
            "Expected `texture_map` to be a tensor. " f"{type(texture_map)=}"
        )
        assert texture_map.ndim == 3 and texture_map.shape[2] == 3, (
            "Expected `texture_map` shape `[H, W, 3]`. " f"{texture_map.shape=}"
        )
        assert texture_map.dtype in {torch.uint8, torch.float32}, (
            "Expected UV texture tensors to be either uint8 or float32. "
            f"{texture_map.dtype=}"
        )

    _validate_inputs()

    def _normalize_inputs() -> np.ndarray:
        if texture_map.dtype == torch.uint8:
            return texture_map.detach().cpu().numpy()

        assert texture_map.dtype == torch.float32, (
            "Expected float texture normalization to receive float32 values. "
            f"{texture_map.dtype=}"
        )
        return (
            texture_map.detach()
            .cpu()
            .clamp(0.0, 1.0)
            .mul(255.0)
            .round()
            .to(torch.uint8)
        ).numpy()

    return _normalize_inputs()


def _rgb_to_css_color(
    rgb_values: np.ndarray,
) -> str:
    """Convert one RGB triplet to a CSS color string.

    Args:
        rgb_values: RGB triplet with shape `(3,)`.

    Returns:
        CSS color string in `rgb(r,g,b)` form.
    """

    def _validate_inputs() -> None:
        assert isinstance(rgb_values, np.ndarray), (
            "Expected `rgb_values` to be a numpy array. " f"{type(rgb_values)=}"
        )
        assert rgb_values.shape == (3,), (
            "Expected one RGB triplet. " f"{rgb_values.shape=}"
        )

    _validate_inputs()

    def _normalize_inputs() -> np.ndarray:
        assert np.issubdtype(rgb_values.dtype, np.integer) or np.issubdtype(
            rgb_values.dtype, np.floating
        ), ("Expected RGB values to be numeric. " f"{rgb_values.dtype=}")
        return np.clip(rgb_values, a_min=0, a_max=255).astype(np.uint8)

    rgb_uint8 = _normalize_inputs()
    return f"rgb({int(rgb_uint8[0])},{int(rgb_uint8[1])},{int(rgb_uint8[2])})"


def _build_textured_triangle_buffers(
    verts: np.ndarray,
    faces: np.ndarray,
    verts_uvs: np.ndarray,
    faces_uvs: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build exploded triangle buffers for one UV-textured mesh.

    Args:
        verts: Mesh verts with shape `(V, 3)`.
        faces: Triangle vertex indices with shape `(F, 3)`.
        verts_uvs: UV coordinate table with shape `(T, 2)`.
        faces_uvs: Triangle UV indices with shape `(F, 3)`.

    Returns:
        Flattened position buffer and flattened UV buffer.
    """

    def _validate_inputs() -> None:
        assert isinstance(verts, np.ndarray), (
            "Expected `verts` to be a numpy array. " f"{type(verts)=}"
        )
        assert verts.ndim == 2 and verts.shape[1] == 3, (
            "Expected `verts` shape `(V, 3)`. " f"{verts.shape=}"
        )
        assert np.issubdtype(verts.dtype, np.floating), (
            "Expected floating-point mesh verts. " f"{verts.dtype=}"
        )

        assert isinstance(faces, np.ndarray), (
            "Expected `faces` to be a numpy array. " f"{type(faces)=}"
        )
        assert faces.ndim == 2 and faces.shape[1] == 3, (
            "Expected `faces` shape `(F, 3)`. " f"{faces.shape=}"
        )
        assert np.issubdtype(faces.dtype, np.integer), (
            "Expected integer face indices. " f"{faces.dtype=}"
        )

        assert isinstance(verts_uvs, np.ndarray), (
            "Expected `verts_uvs` to be a numpy array. " f"{type(verts_uvs)=}"
        )
        assert verts_uvs.ndim == 2 and verts_uvs.shape[1] == 2, (
            "Expected `verts_uvs` shape `(T, 2)`. " f"{verts_uvs.shape=}"
        )
        assert np.issubdtype(verts_uvs.dtype, np.floating), (
            "Expected floating-point UV coordinates. " f"{verts_uvs.dtype=}"
        )

        assert isinstance(faces_uvs, np.ndarray), (
            "Expected `faces_uvs` to be a numpy array. " f"{type(faces_uvs)=}"
        )
        assert faces_uvs.ndim == 2 and faces_uvs.shape[1] == 3, (
            "Expected `faces_uvs` shape `(F, 3)`. " f"{faces_uvs.shape=}"
        )
        assert np.issubdtype(faces_uvs.dtype, np.integer), (
            "Expected integer UV-face indices. " f"{faces_uvs.dtype=}"
        )
        assert faces.shape[0] == faces_uvs.shape[0], (
            "Expected mesh faces and UV faces to have matching triangle counts. "
            f"{faces.shape=} {faces_uvs.shape=}"
        )
        assert int(faces.min()) >= 0, "Expected non-negative face indices. " f"{faces=}"
        assert int(faces.max()) < verts.shape[0], (
            "Expected face indices to address `verts`. "
            f"{faces.max()=} {verts.shape=}"
        )
        assert int(faces_uvs.min()) >= 0, (
            "Expected non-negative UV-face indices. " f"{faces_uvs=}"
        )
        assert int(faces_uvs.max()) < verts_uvs.shape[0], (
            "Expected UV-face indices to address `verts_uvs`. "
            f"{faces_uvs.max()=} {verts_uvs.shape=}"
        )

    _validate_inputs()

    def _normalize_inputs() -> Tuple[np.ndarray, np.ndarray]:
        position_values = verts[faces.reshape(-1)].reshape(-1)
        uv_values = verts_uvs[faces_uvs.reshape(-1)].reshape(-1)
        return position_values, uv_values

    return _normalize_inputs()


def _build_texture_data_url(
    texture_map: np.ndarray,
) -> str:
    """Encode one texture map as an inline PNG data URL.

    Args:
        texture_map: Texture map with shape `(H, W, 3)` and dtype `uint8`.

    Returns:
        PNG data URL string.
    """

    def _validate_inputs() -> None:
        assert isinstance(texture_map, np.ndarray), (
            "Expected `texture_map` to be a numpy array. " f"{type(texture_map)=}"
        )
        assert texture_map.ndim == 3 and texture_map.shape[2] == 3, (
            "Expected `texture_map` shape `(H, W, 3)`. " f"{texture_map.shape=}"
        )
        assert texture_map.dtype == np.uint8, (
            "Expected `texture_map` dtype `uint8`. " f"{texture_map.dtype=}"
        )
        assert texture_map.shape[0] > 0 and texture_map.shape[1] > 0, (
            "Expected a non-empty texture map. " f"{texture_map.shape=}"
        )

    _validate_inputs()

    texture_image = Image.fromarray(texture_map, mode="RGB")
    texture_buffer = io.BytesIO()
    texture_image.save(texture_buffer, format="PNG")
    texture_base64 = base64.b64encode(texture_buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{texture_base64}"


def _build_textured_mesh_html(
    title: str,
    position_values: List[float],
    uv_values: List[float],
    texture_data_url: str,
    mesh_view_bounds: Dict[str, object],
    viewer_id: str,
    camera_sync_group: Optional[str],
) -> str:
    """Build iframe HTML that renders one textured mesh with Three.js.

    Args:
        title: Display title.
        position_values: Flattened triangle positions.
        uv_values: Flattened triangle UV coordinates.
        texture_data_url: Inline PNG texture data URL.
        mesh_view_bounds: Renderer framing summary derived from the mesh bounds.
        viewer_id: Unique iframe viewer id.
        camera_sync_group: Optional browser-side camera-sync group id.

    Returns:
        Standalone iframe HTML document string.
    """

    def _validate_inputs() -> None:
        assert isinstance(title, str), (
            "Expected `title` to be a string. " f"{type(title)=}"
        )
        assert title != "", "Expected `title` to be non-empty. " f"{title=}"

        assert isinstance(position_values, list), (
            "Expected `position_values` to be a list. " f"{type(position_values)=}"
        )
        assert len(position_values) > 0, (
            "Expected non-empty `position_values`. " f"{len(position_values)=}"
        )

        assert isinstance(uv_values, list), (
            "Expected `uv_values` to be a list. " f"{type(uv_values)=}"
        )
        assert len(uv_values) > 0, (
            "Expected non-empty `uv_values`. " f"{len(uv_values)=}"
        )
        assert len(position_values) % 3 == 0, (
            "Expected `position_values` length to be divisible by 3. "
            f"{len(position_values)=}"
        )
        assert len(uv_values) % 2 == 0, (
            "Expected `uv_values` length to be divisible by 2. " f"{len(uv_values)=}"
        )
        assert len(position_values) // 3 == len(uv_values) // 2, (
            "Expected position and UV vertex counts to match. "
            f"{len(position_values)=} {len(uv_values)=}"
        )

        assert isinstance(texture_data_url, str), (
            "Expected `texture_data_url` to be a string. " f"{type(texture_data_url)=}"
        )
        assert texture_data_url.startswith("data:image/png;base64,"), (
            "Expected `texture_data_url` to be a PNG data URL. "
            f"{texture_data_url[:32]=}"
        )

        validate_mesh_view_bounds(mesh_view_bounds=mesh_view_bounds)

        assert isinstance(viewer_id, str), (
            "Expected `viewer_id` to be a string. " f"{type(viewer_id)=}"
        )
        assert viewer_id != "", "Expected `viewer_id` to be non-empty. " f"{viewer_id=}"

        assert camera_sync_group is None or isinstance(camera_sync_group, str), (
            "Expected `camera_sync_group` to be `None` or a string. "
            f"{type(camera_sync_group)=}"
        )

    _validate_inputs()

    viewer_script_template = _load_javascript_template(
        template_path=TEXTURED_MESH_VIEWER_SCRIPT_PATH,
    )

    def _build_viewer_script(camera_sync_script: str) -> str:
        """Build one textured-mesh viewer script with optional camera sync.

        Args:
            camera_sync_script: Iframe-local camera sync JavaScript source.

        Returns:
            Textured-mesh viewer JavaScript source.
        """

        assert isinstance(camera_sync_script, str), (
            "Expected `camera_sync_script` to be a string. "
            f"{type(camera_sync_script)=}"
        )
        return (
            viewer_script_template.replace(
                "__POSITION_VALUES_JSON__",
                json.dumps(position_values),
            )
            .replace(
                "__UV_VALUES_JSON__",
                json.dumps(uv_values),
            )
            .replace(
                "__TEXTURE_DATA_URL_JSON__",
                json.dumps(texture_data_url),
            )
            .replace(
                "__MESH_VIEW_BOUNDS_JSON__",
                json.dumps(mesh_view_bounds),
            )
            .replace(
                "__CAMERA_SYNC_SCRIPT__",
                camera_sync_script,
            )
            .replace(
                "__DEFAULT_CAMERA_EYE_Z_JSON__",
                json.dumps(THREEJS_MESH_DEFAULT_CAMERA_EYE_Z),
            )
            .replace(
                "__VIEWER_ID_JSON__",
                json.dumps(viewer_id),
            )
        )

    viewer_script_without_camera_sync = _build_viewer_script(camera_sync_script="")
    create_dash_trackball_camera_controls(
        renderer_controls=viewer_script_without_camera_sync,
    )

    camera_sync_script = build_threejs_camera_sync_script(
        viewer_id=viewer_id,
        camera_sync_group=camera_sync_group,
    )
    viewer_script = _build_viewer_script(camera_sync_script=camera_sync_script)

    return build_threejs_viewer_html(
        title=title,
        viewer_script=viewer_script,
    )


def build_threejs_viewer_html(
    title: str,
    viewer_script: str,
    extra_script_urls: Optional[List[str]] = None,
) -> str:
    """Build one generic Three.js iframe HTML document.

    Args:
        title: Browser document title.
        viewer_script: JavaScript body that initializes the viewer inside `#mesh-root`.
        extra_script_urls: Optional extra JavaScript URLs to load after Three.js.

    Returns:
        Standalone iframe HTML document string.
    """

    def _validate_inputs() -> None:
        assert isinstance(title, str), (
            "Expected `title` to be a string. " f"{type(title)=}"
        )
        assert title != "", "Expected `title` to be non-empty. " f"{title=}"

        assert isinstance(viewer_script, str), (
            "Expected `viewer_script` to be a string. " f"{type(viewer_script)=}"
        )
        assert viewer_script != "", (
            "Expected `viewer_script` to be non-empty. " f"{viewer_script=}"
        )

        assert extra_script_urls is None or isinstance(extra_script_urls, list), (
            "Expected `extra_script_urls` to be `None` or a list. "
            f"{type(extra_script_urls)=}"
        )
        if extra_script_urls is not None:
            assert all(
                isinstance(script_url, str) for script_url in extra_script_urls
            ), (
                "Expected every `extra_script_urls` entry to be a string. "
                f"{extra_script_urls=}"
            )
            assert all(script_url != "" for script_url in extra_script_urls), (
                "Expected every `extra_script_urls` entry to be non-empty. "
                f"{extra_script_urls=}"
            )

    _validate_inputs()

    normalized_extra_script_urls = (
        [] if extra_script_urls is None else extra_script_urls
    )
    extra_script_tags = "\n".join(
        f'    <script src="{script_url}"></script>'
        for script_url in normalized_extra_script_urls
    )
    extra_script_tags_block = ""
    if extra_script_tags != "":
        extra_script_tags_block = f"\n{extra_script_tags}"

    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{title}</title>
    <script src="https://unpkg.com/three@0.128.0/build/three.min.js"></script>{extra_script_tags_block}
    <style>
      html, body {{
        margin: 0;
        width: 100%;
        height: 100%;
        overflow: hidden;
        background: linear-gradient(160deg, #f7fafc 0%, #edf2f7 100%);
      }}
      #mesh-root {{
        width: 100vw;
        height: 100vh;
      }}
      canvas {{
        display: block;
      }}
    </style>
  </head>
  <body>
    <div id="mesh-root"></div>
    <script>
{viewer_script}
    </script>
  </body>
</html>
"""


@lru_cache(maxsize=None)
def _load_javascript_template(
    template_path: Path,
) -> str:
    """Load one JavaScript template file from disk.

    Args:
        template_path: Filesystem path to one JavaScript template.

    Returns:
        Template text.
    """

    def _validate_inputs() -> None:
        assert isinstance(template_path, Path), (
            "Expected `template_path` to be a `Path`. " f"{type(template_path)=}"
        )
        assert template_path.is_file(), (
            "Expected `template_path` to point to an existing file. "
            f"{template_path=}"
        )
        assert template_path.suffix == ".js", (
            "Expected `template_path` to point to a JavaScript file. "
            f"{template_path=}"
        )

    _validate_inputs()

    def _normalize_inputs() -> Path:
        return template_path.resolve()

    normalized_template_path = _normalize_inputs()

    return normalized_template_path.read_text(encoding="utf-8")


def _apply_mesh_layout(
    figure: go.Figure,
    title: str,
    mesh_view_bounds: Dict[str, object],
) -> None:
    """Apply the shared 3D mesh layout styling to one Plotly figure.

    Args:
        figure: Plotly figure to update in place.
        title: Figure title.
        mesh_view_bounds: Renderer framing summary derived from the mesh bounds.

    Returns:
        None.
    """

    def _validate_inputs() -> None:
        assert isinstance(figure, go.Figure), (
            "Expected `figure` to be a Plotly figure. " f"{type(figure)=}"
        )
        assert isinstance(title, str), (
            "Expected `title` to be a string. " f"{type(title)=}"
        )
        assert title != "", "Expected `title` to be non-empty. " f"{title=}"

        validate_mesh_view_bounds(mesh_view_bounds=mesh_view_bounds)

    _validate_inputs()

    axis_ranges = mesh_view_bounds["axis_ranges"]
    assert isinstance(axis_ranges, dict), f"{axis_ranges=}"

    figure.update_layout(
        meta={"meshViewBounds": mesh_view_bounds},
        title=title,
        margin={"l": 0, "r": 0, "t": 48, "b": 0},
        paper_bgcolor="#ffffff",
        scene={
            "bgcolor": "#f5f7fb",
            "aspectmode": "data",
            "camera": {
                "eye": {"x": 0.0, "y": 0.0, "z": PLOTLY_MESH_DEFAULT_CAMERA_EYE_Z},
                "up": {"x": 0.0, "y": 1.0, "z": 0.0},
            },
            "xaxis": {"visible": False, "range": axis_ranges["x"]},
            "yaxis": {"visible": False, "range": axis_ranges["y"]},
            "zaxis": {"visible": False, "range": axis_ranges["z"]},
        },
        showlegend=False,
        uirevision=PLOTLY_MESH_CAMERA_UIREVISION,
    )


def create_dash_mesh_display(
    mesh: Any,
    mesh_color: Optional[str] = None,
    mesh_opacity: Optional[float] = None,
    mesh_side: Optional[str] = None,
) -> dcc.Graph:
    """Render a Dash mesh display element.

    The `mesh_color`, `mesh_opacity`, and `mesh_side` overrides are opt-in.
    When supplied, `mesh_color` replaces the mesh's texture/per-vertex colors
    with a uniform color so the consumer can override the rendered look without
    rebuilding the mesh data.

    Args:
        mesh: `Mesh` to render; carries either a `MeshTextureVertexColor` or a
            `MeshTextureUVTextureMap` texture.
        mesh_color: Optional uniform color override (CSS color string); when
            None the mesh's texture/per-vertex colors or the lib default color
            is used.
        mesh_opacity: Optional opacity override in `[0, 1]`; when None
            `DEFAULT_MESH_OPACITY` is used.
        mesh_side: Optional side mode override; when None `DEFAULT_MESH_SIDE`
            is used.

    Returns:
        Dash `dcc.Graph` wrapping the mesh scene.
    """
    assert isinstance(mesh, Mesh), (
        "Expected `mesh` to be a `Mesh` instance. " f"{type(mesh)=}"
    )
    assert mesh_color is None or isinstance(mesh_color, str), (
        "Expected `mesh_color` to be None or a CSS color string. "
        f"{type(mesh_color)=}"
    )
    assert mesh_opacity is None or isinstance(mesh_opacity, (int, float)), (
        "Expected `mesh_opacity` to be None or numeric. " f"{type(mesh_opacity)=}"
    )
    assert mesh_side is None or isinstance(mesh_side, str), (
        "Expected `mesh_side` to be None or a string. " f"{type(mesh_side)=}"
    )

    scene = create_dash_mesh_scene(
        mesh=mesh,
        mesh_color=mesh_color,
        mesh_opacity=mesh_opacity,
        mesh_side=mesh_side,
    )
    controls = create_dash_trackball_camera_controls
    return create_dash_mesh_component(
        scene=scene,
        controls=controls,
    )


def create_dash_mesh_scene(
    mesh: Any,
    mesh_color: Optional[str] = None,
    mesh_opacity: Optional[float] = None,
    mesh_side: Optional[str] = None,
) -> go.Mesh3d:
    """Sync-build the Plotly Mesh3d trace from the mesh.

    Args:
        mesh: `Mesh` to render; carries either a `MeshTextureVertexColor` or a
            `MeshTextureUVTextureMap` texture.
        mesh_color: Optional uniform color override (CSS color string); when
            None the mesh's texture/per-vertex colors or `DEFAULT_MESH_COLOR`
            is used.
        mesh_opacity: Optional opacity override in `[0, 1]`; when None
            `DEFAULT_MESH_OPACITY` is used.
        mesh_side: Optional side mode override; when None `DEFAULT_MESH_SIDE`
            is used.

    Returns:
        Plotly `go.Mesh3d` trace for the mesh.
    """
    assert isinstance(mesh, Mesh), (
        "Expected `mesh` to be a `Mesh` instance. " f"{type(mesh)=}"
    )
    assert mesh_color is None or isinstance(mesh_color, str), (
        "Expected `mesh_color` to be None or a CSS color string. "
        f"{type(mesh_color)=}"
    )
    assert mesh_opacity is None or isinstance(mesh_opacity, (int, float)), (
        "Expected `mesh_opacity` to be None or numeric. " f"{type(mesh_opacity)=}"
    )
    assert mesh_side is None or isinstance(mesh_side, str), (
        "Expected `mesh_side` to be None or a string. " f"{type(mesh_side)=}"
    )

    effective_opacity = (
        mesh_opacity if mesh_opacity is not None else DEFAULT_MESH_OPACITY
    )
    effective_side = mesh_side if mesh_side is not None else DEFAULT_MESH_SIDE

    if isinstance(mesh.texture, MeshTextureVertexColor):
        return _create_dash_vertex_color_mesh_scene(
            mesh=mesh,
            mesh_color=mesh_color,
            effective_opacity=effective_opacity,
            effective_side=effective_side,
        )
    if isinstance(mesh.texture, MeshTextureUVTextureMap):
        return _create_dash_uv_texture_map_mesh_scene(
            mesh=mesh,
            mesh_color=mesh_color,
            effective_opacity=effective_opacity,
            effective_side=effective_side,
        )
    raise ValueError(
        "Unsupported mesh texture representation. " f"{type(mesh.texture)=}"
    )


def _create_dash_vertex_color_mesh_scene(
    mesh: Any,
    mesh_color: Optional[str],
    effective_opacity: float,
    effective_side: str,
) -> go.Mesh3d:
    """Sync-build the Plotly Mesh3d trace from a vertex-colored mesh.

    Args:
        mesh: `Mesh` carrying a `MeshTextureVertexColor` texture.
        mesh_color: Optional uniform color override (CSS color string); when
            None the mesh's per-vertex colors or `DEFAULT_MESH_COLOR` is used.
        effective_opacity: Resolved opacity in `[0, 1]`.
        effective_side: Resolved side mode.

    Returns:
        Plotly `go.Mesh3d` trace for the vertex-colored mesh.
    """
    assert isinstance(mesh, Mesh), (
        "Expected `mesh` to be a `Mesh` instance. " f"{type(mesh)=}"
    )
    assert isinstance(mesh.texture, MeshTextureVertexColor), (
        "Expected `mesh` to carry a `MeshTextureVertexColor` texture. "
        f"{type(mesh.texture)=}"
    )
    assert mesh_color is None or isinstance(mesh_color, str), (
        "Expected `mesh_color` to be None or a CSS color string. "
        f"{type(mesh_color)=}"
    )
    assert isinstance(effective_opacity, (int, float)), (
        "Expected `effective_opacity` to be numeric. " f"{type(effective_opacity)=}"
    )
    assert isinstance(effective_side, str), (
        "Expected `effective_side` to be a string. " f"{type(effective_side)=}"
    )

    verts_np = mesh.verts.detach().cpu().numpy()
    faces_np = mesh.faces.detach().cpu().numpy()

    if mesh_color is not None:
        effective_color: Union[str, List[str]] = mesh_color
    else:
        vertex_colors = _normalize_rgb_tensor_to_uint8(
            rgb_values=mesh.texture.vertex_color,
        )
        effective_color = [
            _rgb_to_css_color(rgb_values=vertex_rgb) for vertex_rgb in vertex_colors
        ]

    trace_kwargs: Dict[str, Any] = dict(
        x=verts_np[:, 0],
        y=verts_np[:, 1],
        z=verts_np[:, 2],
        i=faces_np[:, 0],
        j=faces_np[:, 1],
        k=faces_np[:, 2],
        opacity=effective_opacity,
        flatshading=False,
        hoverinfo="skip",
    )
    if isinstance(effective_color, list):
        trace_kwargs["vertexcolor"] = effective_color
    else:
        trace_kwargs["color"] = effective_color
    return go.Mesh3d(**trace_kwargs)


def _create_dash_uv_texture_map_mesh_scene(
    mesh: Any,
    mesh_color: Optional[str],
    effective_opacity: float,
    effective_side: str,
) -> go.Mesh3d:
    """Sync-build the Plotly Mesh3d trace from a UV-textured mesh.

    Args:
        mesh: `Mesh` carrying a `MeshTextureUVTextureMap` texture.
        mesh_color: Optional uniform color override (CSS color string); when
            None the mesh's UV-texture-sampled colors or `DEFAULT_MESH_COLOR`
            is used.
        effective_opacity: Resolved opacity in `[0, 1]`.
        effective_side: Resolved side mode.

    Returns:
        Plotly `go.Mesh3d` trace for the UV-textured mesh.
    """
    assert isinstance(mesh, Mesh), (
        "Expected `mesh` to be a `Mesh` instance. " f"{type(mesh)=}"
    )
    assert isinstance(mesh.texture, MeshTextureUVTextureMap), (
        "Expected `mesh` to carry a `MeshTextureUVTextureMap` texture. "
        f"{type(mesh.texture)=}"
    )
    assert mesh_color is None or isinstance(mesh_color, str), (
        "Expected `mesh_color` to be None or a CSS color string. "
        f"{type(mesh_color)=}"
    )
    assert isinstance(effective_opacity, (int, float)), (
        "Expected `effective_opacity` to be numeric. " f"{type(effective_opacity)=}"
    )
    assert isinstance(effective_side, str), (
        "Expected `effective_side` to be a string. " f"{type(effective_side)=}"
    )

    verts_np = mesh.verts.detach().cpu().numpy()
    faces_np = mesh.faces.detach().cpu().numpy()

    if mesh_color is not None:
        effective_color: Union[str, List[str]] = mesh_color
    else:
        texture_map = _normalize_texture_map_to_uint8(
            texture_map=mesh.texture.uv_texture_map,
        )
        texture_height, texture_width = texture_map.shape[0], texture_map.shape[1]
        verts_uvs_np = mesh.texture.verts_uvs.detach().cpu().numpy()
        faces_uvs_np = mesh.texture.faces_uvs.detach().cpu().numpy()
        vertex_uvs = np.zeros((mesh.verts.shape[0], 2), dtype=verts_uvs_np.dtype)
        vertex_uvs[faces_np.reshape(-1)] = verts_uvs_np[faces_uvs_np.reshape(-1)]
        column = np.clip(
            np.round(vertex_uvs[:, 0] * (texture_width - 1)).astype(np.int64),
            a_min=0,
            a_max=texture_width - 1,
        )
        row = np.clip(
            np.round((1.0 - vertex_uvs[:, 1]) * (texture_height - 1)).astype(np.int64),
            a_min=0,
            a_max=texture_height - 1,
        )
        sampled_rgb = texture_map[row, column]
        effective_color = [
            _rgb_to_css_color(rgb_values=vertex_rgb) for vertex_rgb in sampled_rgb
        ]

    trace_kwargs: Dict[str, Any] = dict(
        x=verts_np[:, 0],
        y=verts_np[:, 1],
        z=verts_np[:, 2],
        i=faces_np[:, 0],
        j=faces_np[:, 1],
        k=faces_np[:, 2],
        opacity=effective_opacity,
        flatshading=False,
        hoverinfo="skip",
    )
    if isinstance(effective_color, list):
        trace_kwargs["vertexcolor"] = effective_color
    else:
        trace_kwargs["color"] = effective_color
    return go.Mesh3d(**trace_kwargs)


def create_dash_mesh_component(
    scene: go.Mesh3d,
    controls: Any,
) -> dcc.Graph:
    """Wrap the mesh scene and camera controls into a Dash component.

    Args:
        scene: Plotly `go.Mesh3d` trace for the mesh.
        controls: Dash trackball camera-controls factory for the renderer.

    Returns:
        Dash `dcc.Graph` rendering the mesh scene.
    """
    assert isinstance(scene, go.Mesh3d), (
        "Expected `scene` to be a Plotly `go.Mesh3d` trace. " f"{type(scene)=}"
    )
    return dcc.Graph(figure=go.Figure(data=[scene]))
