"""Dash mesh display APIs."""

from typing import Optional

import torch
from dash import dcc

from data.structures.three_d.mesh.mesh import Mesh
from data.structures.three_d.mesh.texture.mesh_texture_uv_texture_map import (
    MeshTextureUVTextureMap,
)
from data.structures.three_d.mesh.texture.mesh_texture_vertex_color import (
    MeshTextureVertexColor,
)
from data.viewer.utils.atomic_displays.mesh.dash.core_mesh_display import (
    create_dash_mesh_display,
    create_mesh_display,
)
from data.viewer.utils.atomic_displays.utils.class_colors import map_class_ids_to_rgb
from data.viewer.utils.atomic_displays.utils.heatmap_colors import map_scalars_to_rgb

__all__ = [
    "create_color_mesh_display",
    "create_heatmap_mesh_display",
    "create_mesh_display",
    "create_segmentation_mesh_display",
]


def create_color_mesh_display(
    color_mesh_path: str,
    mesh_color: Optional[str] = None,
    mesh_opacity: Optional[float] = None,
    mesh_side: Optional[str] = None,
) -> dcc.Graph:
    """Render a color mesh display from a color mesh artifact path.

    Args:
        color_mesh_path: Color mesh artifact path on disk.
        mesh_color: Optional uniform color override (CSS color string); when
            None the mesh's texture/per-vertex colors or the lib default color
            is used.
        mesh_opacity: Optional opacity override in `[0, 1]`; when None the lib
            default opacity is used.
        mesh_side: Optional side mode override; when None the lib default side
            mode is used.

    Returns:
        Dash `dcc.Graph` wrapping the color mesh scene.
    """
    assert isinstance(color_mesh_path, str), (
        "Expected `color_mesh_path` to be a string. " f"{type(color_mesh_path)=}"
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

    mesh = Mesh.load(path=color_mesh_path)
    return create_dash_mesh_display(
        mesh=mesh,
        mesh_color=mesh_color,
        mesh_opacity=mesh_opacity,
        mesh_side=mesh_side,
    )


def create_segmentation_mesh_display(
    segmentation_mesh_path: str,
    mesh_opacity: Optional[float] = None,
    mesh_side: Optional[str] = None,
) -> dcc.Graph:
    """Render a backend-colorized segmentation mesh display.

    Per-element colors are baked in by the backend's class-id -> rgb mapping,
    so no `mesh_color` override is exposed here.

    Args:
        segmentation_mesh_path: Class-labeled segmentation mesh artifact path
            on disk.
        mesh_opacity: Optional opacity override in `[0, 1]`; when None the lib
            default opacity is used.
        mesh_side: Optional side mode override; when None the lib default side
            mode is used.

    Returns:
        Dash `dcc.Graph` wrapping the colorized segmentation mesh scene.
    """
    assert isinstance(segmentation_mesh_path, str), (
        "Expected `segmentation_mesh_path` to be a string. "
        f"{type(segmentation_mesh_path)=}"
    )
    assert mesh_opacity is None or isinstance(mesh_opacity, (int, float)), (
        "Expected `mesh_opacity` to be None or numeric. " f"{type(mesh_opacity)=}"
    )
    assert mesh_side is None or isinstance(mesh_side, str), (
        "Expected `mesh_side` to be None or a string. " f"{type(mesh_side)=}"
    )

    segmentation_mesh = Mesh.load(path=segmentation_mesh_path)
    if isinstance(segmentation_mesh.texture, MeshTextureVertexColor):
        segmentation_mesh_class_ids = segmentation_mesh.texture.vertex_color[:, 0].to(
            dtype=torch.int64
        )
    elif isinstance(segmentation_mesh.texture, MeshTextureUVTextureMap):
        segmentation_mesh_class_ids = segmentation_mesh.texture.uv_texture_map[
            ..., 0
        ].to(dtype=torch.int64)
    else:
        raise ValueError(
            "Unsupported segmentation mesh texture representation. "
            f"{type(segmentation_mesh.texture)=}"
        )
    class_id_to_rgb = map_class_ids_to_rgb(
        class_ids=torch.unique(segmentation_mesh_class_ids),
    )
    colorized_mesh = _map_segmentation_mesh_to_rgb(
        mesh=segmentation_mesh,
        class_id_to_rgb=class_id_to_rgb,
    )
    return create_dash_mesh_display(
        mesh=colorized_mesh,
        mesh_opacity=mesh_opacity,
        mesh_side=mesh_side,
    )


def create_heatmap_mesh_display(
    heatmap_mesh_path: str,
    mesh_opacity: Optional[float] = None,
    mesh_side: Optional[str] = None,
) -> dcc.Graph:
    """Render a backend-colorized heatmap mesh display.

    Per-element colors are baked in by the backend's scalar -> rgb mapping, so
    no `mesh_color` override is exposed here.

    Args:
        heatmap_mesh_path: Non-negative-scalar-labeled mesh artifact path on
            disk.
        mesh_opacity: Optional opacity override in `[0, 1]`; when None the lib
            default opacity is used.
        mesh_side: Optional side mode override; when None the lib default side
            mode is used.

    Returns:
        Dash `dcc.Graph` wrapping the colorized heatmap mesh scene.
    """
    assert isinstance(heatmap_mesh_path, str), (
        "Expected `heatmap_mesh_path` to be a string. " f"{type(heatmap_mesh_path)=}"
    )
    assert mesh_opacity is None or isinstance(mesh_opacity, (int, float)), (
        "Expected `mesh_opacity` to be None or numeric. " f"{type(mesh_opacity)=}"
    )
    assert mesh_side is None or isinstance(mesh_side, str), (
        "Expected `mesh_side` to be None or a string. " f"{type(mesh_side)=}"
    )

    heatmap_mesh = Mesh.load(path=heatmap_mesh_path)
    if isinstance(heatmap_mesh.texture, MeshTextureVertexColor):
        heatmap_mesh_scalars = heatmap_mesh.texture.vertex_color[:, 0]
    elif isinstance(heatmap_mesh.texture, MeshTextureUVTextureMap):
        heatmap_mesh_scalars = heatmap_mesh.texture.uv_texture_map[..., 0]
    else:
        raise ValueError(
            "Unsupported heatmap mesh texture representation. "
            f"{type(heatmap_mesh.texture)=}"
        )
    scalar_rgb = map_scalars_to_rgb(scalars=heatmap_mesh_scalars)
    colorized_mesh = _map_heatmap_mesh_to_rgb(
        mesh=heatmap_mesh,
        scalar_rgb=scalar_rgb,
    )
    return create_dash_mesh_display(
        mesh=colorized_mesh,
        mesh_opacity=mesh_opacity,
        mesh_side=mesh_side,
    )


def _map_segmentation_mesh_to_rgb(
    mesh: Mesh,
    class_id_to_rgb: dict,
) -> Mesh:
    """Apply `class_id_to_rgb` to the segmentation mesh's class-id storage.

    Args:
        mesh: Class-labeled segmentation mesh with a `MeshTextureVertexColor`
            (per-vertex) or a `MeshTextureUVTextureMap` (per-texel) texture.
        class_id_to_rgb: Mapping from class id to RGB color tuple with channel
            values in `[0, 255]`.

    Returns:
        Colored `Mesh` with the same geometry and class-id storage replaced by
        per-element RGB.
    """
    assert isinstance(mesh, Mesh), (
        "Expected `mesh` to be a `Mesh` instance. " f"{type(mesh)=}"
    )
    assert isinstance(class_id_to_rgb, dict), (
        "Expected `class_id_to_rgb` to be a `dict`. " f"{type(class_id_to_rgb)=}"
    )

    if isinstance(mesh.texture, MeshTextureVertexColor):
        class_ids = mesh.texture.vertex_color[:, 0].to(dtype=torch.int64)
        vertex_color = torch.zeros((class_ids.shape[0], 3), dtype=torch.uint8)
        for class_id, color in class_id_to_rgb.items():
            vertex_color[class_ids == int(class_id)] = torch.tensor(
                color,
                dtype=torch.uint8,
            )
        return Mesh(
            verts=mesh.verts,
            faces=mesh.faces,
            texture=MeshTextureVertexColor(vertex_color=vertex_color),
        )
    if isinstance(mesh.texture, MeshTextureUVTextureMap):
        class_ids = mesh.texture.uv_texture_map[..., 0].to(dtype=torch.int64)
        height, width = class_ids.shape
        uv_texture_map = torch.zeros((height, width, 3), dtype=torch.uint8)
        for class_id, color in class_id_to_rgb.items():
            uv_texture_map[class_ids == int(class_id)] = torch.tensor(
                color,
                dtype=torch.uint8,
            )
        return Mesh(
            verts=mesh.verts,
            faces=mesh.faces,
            texture=MeshTextureUVTextureMap(
                uv_texture_map=uv_texture_map,
                verts_uvs=mesh.texture.verts_uvs,
                faces_uvs=mesh.texture.faces_uvs,
                convention=mesh.texture.convention,
            ),
        )
    raise ValueError(
        "Unsupported segmentation mesh texture representation. "
        f"{type(mesh.texture)=}"
    )


def _map_heatmap_mesh_to_rgb(
    mesh: Mesh,
    scalar_rgb: torch.Tensor,
) -> Mesh:
    """Write `scalar_rgb` onto the heatmap mesh's scalar storage.

    Args:
        mesh: Non-negative-scalar-labeled mesh with a `MeshTextureVertexColor`
            (per-vertex) or a `MeshTextureUVTextureMap` (per-texel) texture.
        scalar_rgb: Colorized scalar tensor of shape `scalars.shape + (3,)`,
            uint8 dtype in `[0, 255]`.

    Returns:
        Colored `Mesh` with the same geometry and scalar storage replaced by
        per-element RGB.
    """
    assert isinstance(mesh, Mesh), (
        "Expected `mesh` to be a `Mesh` instance. " f"{type(mesh)=}"
    )
    assert isinstance(scalar_rgb, torch.Tensor), (
        "Expected `scalar_rgb` to be a `torch.Tensor`. " f"{type(scalar_rgb)=}"
    )
    assert scalar_rgb.dtype == torch.uint8, (
        "Expected `scalar_rgb` to be uint8. " f"{scalar_rgb.dtype=}"
    )

    if isinstance(mesh.texture, MeshTextureVertexColor):
        assert scalar_rgb.ndim == 2 and scalar_rgb.shape[-1] == 3, (
            "Expected per-vertex `scalar_rgb` to be `[V, 3]`. " f"{scalar_rgb.shape=}"
        )
        return Mesh(
            verts=mesh.verts,
            faces=mesh.faces,
            texture=MeshTextureVertexColor(vertex_color=scalar_rgb),
        )
    if isinstance(mesh.texture, MeshTextureUVTextureMap):
        assert scalar_rgb.ndim == 3 and scalar_rgb.shape[-1] == 3, (
            "Expected per-texel `scalar_rgb` to be `[H, W, 3]`. " f"{scalar_rgb.shape=}"
        )
        return Mesh(
            verts=mesh.verts,
            faces=mesh.faces,
            texture=MeshTextureUVTextureMap(
                uv_texture_map=scalar_rgb,
                verts_uvs=mesh.texture.verts_uvs,
                faces_uvs=mesh.texture.faces_uvs,
                convention=mesh.texture.convention,
            ),
        )
    raise ValueError(
        "Unsupported heatmap mesh texture representation. " f"{type(mesh.texture)=}"
    )
