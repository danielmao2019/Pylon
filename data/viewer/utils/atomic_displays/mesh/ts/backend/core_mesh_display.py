"""Mesh display response core."""

from pathlib import Path
from typing import Any, Dict

from data.structures.three_d.mesh.load import load_mesh
from data.structures.three_d.mesh.mesh import Mesh
from data.structures.three_d.mesh.save import save_mesh
from data.structures.three_d.mesh.texture.mesh_texture_uv_texture_map import (
    MeshTextureUVTextureMap,
)
from data.structures.three_d.mesh.texture.mesh_texture_vertex_color import (
    MeshTextureVertexColor,
)
from data.viewer.utils.atomic_displays.mesh.ts.backend.schemas.display_response import (
    MeshDisplayResponse,
)


def create_mesh_display_response(
    input_path: Path,
    output_path: Path,
    url: str,
    slot_id: str,
    title: str,
    meta_info: Dict[str, Any],
) -> MeshDisplayResponse:
    """Create a mesh display response.

    Args:
        input_path: Input mesh artifact path on disk.
        output_path: Output mesh artifact path on disk; the processed mesh
            resource bytes are written here.
        url: Frontend resource URL pointing at the written output mesh.
        slot_id: Stable display slot identifier.
        title: Display panel title.
        meta_info: Caller-provided renderer metadata.

    Returns:
        Mesh display response with the caller-provided url/slot_id/title/meta_info.
    """
    assert isinstance(
        input_path, Path
    ), "Expected `input_path` to be a `Path`. input_path=%r" % (input_path,)
    assert isinstance(
        output_path, Path
    ), "Expected `output_path` to be a `Path`. output_path=%r" % (output_path,)
    assert isinstance(url, str), "Expected `url` to be a `str`. url=%r" % (url,)
    assert isinstance(slot_id, str), "Expected `slot_id` to be a `str`. slot_id=%r" % (
        slot_id,
    )
    assert isinstance(title, str), "Expected `title` to be a `str`. title=%r" % (title,)
    assert isinstance(
        meta_info, dict
    ), "Expected `meta_info` to be a `dict`. meta_info=%r" % (meta_info,)

    mesh = load_mesh(path=input_path)
    if isinstance(mesh.texture, MeshTextureVertexColor):
        _create_vertex_color_mesh_display(mesh=mesh, output_path=output_path)
    elif isinstance(mesh.texture, MeshTextureUVTextureMap):
        _create_uv_texture_map_mesh_display(mesh=mesh, output_path=output_path)
    else:
        raise ValueError(
            "Unsupported mesh texture representation. texture=%r" % (mesh.texture,),
        )
    return MeshDisplayResponse(
        slot_id=slot_id,
        title=title,
        url=url,
        meta_info=dict(meta_info),
    )


def _create_vertex_color_mesh_display(mesh: Mesh, output_path: Path) -> None:
    """Write a vertex-colored mesh resource to disk.

    Args:
        mesh: Loaded mesh with per-vertex color storage.
        output_path: Output mesh artifact path on disk.

    Returns:
        None.
    """
    assert isinstance(mesh, Mesh), "Expected `mesh` to be a `Mesh`. mesh=%r" % (mesh,)
    assert isinstance(
        output_path, Path
    ), "Expected `output_path` to be a `Path`. output_path=%r" % (output_path,)
    assert isinstance(mesh.texture, MeshTextureVertexColor), (
        "Expected vertex-color mesh display to receive a mesh with a "
        "MeshTextureVertexColor. mesh.texture=%r" % (mesh.texture,)
    )

    save_mesh(mesh=mesh, output_path=output_path)


def _create_uv_texture_map_mesh_display(mesh: Mesh, output_path: Path) -> None:
    """Write a UV-textured mesh resource to disk.

    Args:
        mesh: Loaded mesh with UV texture map storage.
        output_path: Output mesh artifact path on disk.

    Returns:
        None.
    """
    assert isinstance(mesh, Mesh), "Expected `mesh` to be a `Mesh`. mesh=%r" % (mesh,)
    assert isinstance(
        output_path, Path
    ), "Expected `output_path` to be a `Path`. output_path=%r" % (output_path,)
    assert isinstance(mesh.texture, MeshTextureUVTextureMap), (
        "Expected UV-textured mesh display to receive a mesh with a "
        "MeshTextureUVTextureMap. mesh.texture=%r" % (mesh.texture,)
    )

    save_mesh(mesh=mesh, output_path=output_path)
