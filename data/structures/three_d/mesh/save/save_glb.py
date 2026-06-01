from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import torch

from data.structures.three_d.mesh.mesh import Mesh
from data.structures.three_d.mesh.texture.canonicalize import (
    collapse_seam_shifted_uv_rows,
)
from data.structures.three_d.mesh.texture.mesh_texture_uv_texture_map import (
    MeshTextureUVTextureMap,
)
from data.structures.three_d.mesh.texture.mesh_texture_vertex_color import (
    MeshTextureVertexColor,
)
from utils.io.glb import append_accessor, append_image, write_glb
from utils.io.image import encode_image_bytes

_ARRAY_BUFFER: int = 34962
_ELEMENT_ARRAY_BUFFER: int = 34963


def save_glb_mesh(mesh: Mesh, output_path: Union[str, Path]) -> None:
    """Write a Mesh to a GLB container, dispatched to the texture-representation-specific writer.

    Args:
        mesh: `Mesh` instance to save.
        output_path: Output GLB filepath or output directory path.

    Returns:
        None.
    """

    def _validate_inputs() -> None:
        assert isinstance(mesh, Mesh), (
            "Expected `mesh` to be a `Mesh` instance. " f"{type(mesh)=}"
        )

    _validate_inputs()

    glb_path = _resolve_output_glb_path(output_path=output_path)
    glb_path.parent.mkdir(parents=True, exist_ok=True)

    if mesh.texture is None:
        _save_geometry_only_glb(mesh=mesh, glb_path=glb_path)
        return
    if isinstance(mesh.texture, MeshTextureVertexColor):
        _save_vertex_color_glb(mesh=mesh, glb_path=glb_path)
        return
    if isinstance(mesh.texture, MeshTextureUVTextureMap):
        _save_uv_texture_map_glb(mesh=mesh, glb_path=glb_path)
        return
    assert 0, (
        "should not reach here: a Mesh texture is None, MeshTextureVertexColor, "
        f"or MeshTextureUVTextureMap. {type(mesh.texture)=}"
    )


def _save_geometry_only_glb(mesh: Mesh, glb_path: Path) -> None:
    """Append POSITION + indices accessors and write the GLB (no material).

    Args:
        mesh: Geometry-only `Mesh`.
        glb_path: Concrete GLB output filepath.

    Returns:
        None.
    """

    verts = mesh.verts.detach().cpu().numpy().astype(np.float32)
    faces = mesh.faces.detach().cpu().numpy().reshape(-1).astype(np.uint32)
    gltf: Dict[str, Any] = {"asset": {"version": "2.0"}, "buffers": [{}]}
    binary_blob = bytearray()
    position_accessor = append_accessor(
        gltf=gltf, binary_blob=binary_blob, array=verts, target=_ARRAY_BUFFER
    )
    index_accessor = append_accessor(
        gltf=gltf, binary_blob=binary_blob, array=faces, target=_ELEMENT_ARRAY_BUFFER
    )
    gltf["meshes"] = [
        {
            "primitives": [
                {
                    "attributes": {"POSITION": position_accessor},
                    "indices": index_accessor,
                }
            ]
        }
    ]
    gltf["nodes"] = [{"mesh": 0}]
    gltf["scenes"] = [{"nodes": [0]}]
    gltf["scene"] = 0
    write_glb(gltf=gltf, binary_blob=binary_blob, path=glb_path)


def _save_vertex_color_glb(mesh: Mesh, glb_path: Path) -> None:
    """Append POSITION + indices + COLOR_0 accessors and write the GLB (no texture material).

    Args:
        mesh: `Mesh` carrying a `MeshTextureVertexColor`.
        glb_path: Concrete GLB output filepath.

    Returns:
        None.
    """

    verts = mesh.verts.detach().cpu().numpy().astype(np.float32)
    faces = mesh.faces.detach().cpu().numpy().reshape(-1).astype(np.uint32)
    vertex_color = (
        mesh.texture.vertex_color.detach()
        .cpu()
        .reshape(-1, 3)
        .to(torch.float32)
        .numpy()
        .astype(np.float32)
    )
    gltf: Dict[str, Any] = {"asset": {"version": "2.0"}, "buffers": [{}]}
    binary_blob = bytearray()
    position_accessor = append_accessor(
        gltf=gltf, binary_blob=binary_blob, array=verts, target=_ARRAY_BUFFER
    )
    color_accessor = append_accessor(
        gltf=gltf, binary_blob=binary_blob, array=vertex_color, target=_ARRAY_BUFFER
    )
    index_accessor = append_accessor(
        gltf=gltf, binary_blob=binary_blob, array=faces, target=_ELEMENT_ARRAY_BUFFER
    )
    gltf["meshes"] = [
        {
            "primitives": [
                {
                    "attributes": {
                        "POSITION": position_accessor,
                        "COLOR_0": color_accessor,
                    },
                    "indices": index_accessor,
                }
            ]
        }
    ]
    gltf["nodes"] = [{"mesh": 0}]
    gltf["scenes"] = [{"nodes": [0]}]
    gltf["scene"] = 0
    write_glb(gltf=gltf, binary_blob=binary_blob, path=glb_path)


def _save_uv_texture_map_glb(mesh: Mesh, glb_path: Path) -> None:
    """Append POSITION + indices + TEXCOORD_0 accessors + an embedded base-color texture image, then write the GLB.

    The seam-safe canonical UVs are collapsed and scattered onto the geometry
    vertex order so glTF's single shared index addresses both POSITION and
    TEXCOORD_0.

    Args:
        mesh: `Mesh` carrying a `MeshTextureUVTextureMap`.
        glb_path: Concrete GLB output filepath.

    Returns:
        None.
    """

    faces = mesh.faces.detach().cpu()
    collapsed_verts_uvs, collapsed_faces_uvs = collapse_seam_shifted_uv_rows(
        verts_uvs=mesh.texture.verts_uvs.detach().cpu(),
        faces_uvs=mesh.texture.faces_uvs.detach().cpu(),
    )
    per_vertex_uv = torch.zeros((int(mesh.verts.shape[0]), 2), dtype=torch.float32)
    per_vertex_uv[faces.reshape(-1)] = collapsed_verts_uvs[
        collapsed_faces_uvs.reshape(-1)
    ].to(torch.float32)

    texture = mesh.texture.uv_texture_map.detach().cpu()
    if texture.dtype == torch.uint8:
        texture_uint8 = texture
    else:
        texture_uint8 = (texture.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
    image_bytes = encode_image_bytes(image=texture_uint8, image_format="PNG")

    verts = mesh.verts.detach().cpu().numpy().astype(np.float32)
    uv = per_vertex_uv.numpy().astype(np.float32)
    faces_flat = faces.numpy().reshape(-1).astype(np.uint32)

    gltf: Dict[str, Any] = {"asset": {"version": "2.0"}, "buffers": [{}]}
    binary_blob = bytearray()
    position_accessor = append_accessor(
        gltf=gltf, binary_blob=binary_blob, array=verts, target=_ARRAY_BUFFER
    )
    texcoord_accessor = append_accessor(
        gltf=gltf, binary_blob=binary_blob, array=uv, target=_ARRAY_BUFFER
    )
    index_accessor = append_accessor(
        gltf=gltf,
        binary_blob=binary_blob,
        array=faces_flat,
        target=_ELEMENT_ARRAY_BUFFER,
    )
    image_index = append_image(
        gltf=gltf,
        binary_blob=binary_blob,
        image_bytes=image_bytes,
        mime_type="image/png",
    )
    gltf["samplers"] = [{}]
    gltf["textures"] = [{"sampler": 0, "source": image_index}]
    gltf["materials"] = [
        {
            "pbrMetallicRoughness": {
                "baseColorTexture": {"index": 0},
                "metallicFactor": 0.0,
                "roughnessFactor": 1.0,
            }
        }
    ]
    gltf["meshes"] = [
        {
            "primitives": [
                {
                    "attributes": {
                        "POSITION": position_accessor,
                        "TEXCOORD_0": texcoord_accessor,
                    },
                    "indices": index_accessor,
                    "material": 0,
                }
            ]
        }
    ]
    gltf["nodes"] = [{"mesh": 0}]
    gltf["scenes"] = [{"nodes": [0]}]
    gltf["scene"] = 0
    write_glb(gltf=gltf, binary_blob=binary_blob, path=glb_path)


def _resolve_output_glb_path(output_path: Union[str, Path]) -> Path:
    """Resolve an output path to a concrete `.glb` file path.

    Args:
        output_path: Output GLB filepath or output directory path.

    Returns:
        Concrete GLB output filepath (a `.glb` path, or `<dir>/mesh.glb`).
    """

    def _validate_inputs() -> None:
        assert isinstance(output_path, (str, Path)), (
            "Expected `output_path` to be a `str` or `Path`. " f"{type(output_path)=}"
        )

    _validate_inputs()

    candidate_path = Path(output_path)
    if candidate_path.suffix.lower() == ".glb":
        return candidate_path
    assert candidate_path.suffix == "", (
        "Expected GLB mesh saving to target a `.glb` file or a directory-like "
        "path without a suffix. "
        f"{candidate_path=}"
    )
    return candidate_path / "mesh.glb"
