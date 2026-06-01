from pathlib import Path
from typing import Any, Dict, Tuple, Union

import numpy as np
import torch

from data.structures.three_d.mesh.mesh import Mesh
from data.structures.three_d.mesh.texture.canonicalize import (
    shift_seam_crossing_faces_to_seam_safe,
)
from data.structures.three_d.mesh.texture.mesh_texture_uv_texture_map import (
    MeshTextureUVTextureMap,
)
from data.structures.three_d.mesh.texture.mesh_texture_vertex_color import (
    MeshTextureVertexColor,
)
from utils.io.glb import load_glb_json_and_bin, read_accessor, read_image_bytes
from utils.io.image import decode_image_bytes


def load_glb_mesh(path: Union[str, Path]) -> Mesh:
    """Load one GLB file into a Mesh, dispatched by texture representation.

    A GLB is one self-contained file, so there is no block split: one primitive
    is selected and built into a geometry-only, vertex-colored, or UV-textured
    Mesh.

    Args:
        path: Filesystem path to the GLB file.

    Returns:
        One `Mesh`.
    """

    def _validate_inputs() -> None:
        assert isinstance(path, (str, Path)), (
            "Expected `path` to be a `str` or `Path`. " f"{type(path)=}"
        )
        assert Path(path).is_file() and Path(path).suffix.lower() == ".glb", (
            "Expected GLB loading to receive an existing `.glb` file. " f"{path=}"
        )

    _validate_inputs()

    gltf, binary_blob = load_glb_json_and_bin(path=path)
    mesh_index, primitive_index = _select_mesh_primitive(gltf=gltf)
    primitive = gltf["meshes"][mesh_index]["primitives"][primitive_index]
    attributes = primitive.get("attributes", {})
    has_color_0 = "COLOR_0" in attributes
    has_texcoord_0 = "TEXCOORD_0" in attributes

    materials = gltf.get("materials", [])
    material_index = primitive.get("material")
    has_base_color_texture = (
        material_index is not None
        and material_index < len(materials)
        and "baseColorTexture"
        in materials[material_index].get("pbrMetallicRoughness", {})
    )

    if not has_color_0 and not (has_texcoord_0 and has_base_color_texture):
        return _load_glb_geometry_only(
            gltf=gltf,
            binary_blob=binary_blob,
            mesh_index=mesh_index,
            primitive_index=primitive_index,
        )
    if has_color_0:
        return _load_glb_vertex_color(
            gltf=gltf,
            binary_blob=binary_blob,
            mesh_index=mesh_index,
            primitive_index=primitive_index,
        )
    if has_texcoord_0 and has_base_color_texture:
        return _load_glb_uv_texture_map(
            gltf=gltf,
            binary_blob=binary_blob,
            mesh_index=mesh_index,
            primitive_index=primitive_index,
        )
    assert 0, (
        "should not reach here: a GLB primitive is geometry-only, vertex-color, "
        f"or uv-texture-map. {has_color_0=} {has_texcoord_0=} {has_base_color_texture=}"
    )


def _load_glb_geometry_only(
    gltf: Dict[str, Any],
    binary_blob: bytes,
    mesh_index: int,
    primitive_index: int,
) -> Mesh:
    """Build a geometry-only Mesh from a GLB primitive.

    Args:
        gltf: Decoded glTF JSON dictionary.
        binary_blob: GLB BIN chunk bytes.
        mesh_index: Index into ``gltf["meshes"]``.
        primitive_index: Index into the mesh's ``primitives``.

    Returns:
        One geometry-only `Mesh` (texture `None`).
    """

    primitive = gltf["meshes"][mesh_index]["primitives"][primitive_index]
    verts_np = read_accessor(
        gltf=gltf,
        binary_blob=binary_blob,
        accessor_index=primitive["attributes"]["POSITION"],
    )
    faces_np = read_accessor(
        gltf=gltf, binary_blob=binary_blob, accessor_index=primitive["indices"]
    ).reshape(-1, 3)
    return Mesh(
        verts=torch.from_numpy(verts_np.astype(np.float32)).contiguous(),
        faces=torch.from_numpy(faces_np.astype(np.int64)).contiguous(),
        texture=None,
    )


def _load_glb_vertex_color(
    gltf: Dict[str, Any],
    binary_blob: bytes,
    mesh_index: int,
    primitive_index: int,
) -> Mesh:
    """Build a vertex-colored Mesh from a GLB primitive.

    Args:
        gltf: Decoded glTF JSON dictionary.
        binary_blob: GLB BIN chunk bytes.
        mesh_index: Index into ``gltf["meshes"]``.
        primitive_index: Index into the mesh's ``primitives``.

    Returns:
        One `Mesh` carrying a `MeshTextureVertexColor`.
    """

    primitive = gltf["meshes"][mesh_index]["primitives"][primitive_index]
    verts_np = read_accessor(
        gltf=gltf,
        binary_blob=binary_blob,
        accessor_index=primitive["attributes"]["POSITION"],
    )
    faces_np = read_accessor(
        gltf=gltf, binary_blob=binary_blob, accessor_index=primitive["indices"]
    ).reshape(-1, 3)
    color_np = read_accessor(
        gltf=gltf,
        binary_blob=binary_blob,
        accessor_index=primitive["attributes"]["COLOR_0"],
    )
    vertex_color = torch.from_numpy(color_np.astype(np.float32))
    if color_np.dtype == np.uint8:
        vertex_color = vertex_color / 255.0
    elif color_np.dtype == np.uint16:
        vertex_color = vertex_color / 65535.0
    return Mesh(
        verts=torch.from_numpy(verts_np.astype(np.float32)).contiguous(),
        faces=torch.from_numpy(faces_np.astype(np.int64)).contiguous(),
        texture=MeshTextureVertexColor(vertex_color=vertex_color[:, :3].contiguous()),
    )


def _load_glb_uv_texture_map(
    gltf: Dict[str, Any],
    binary_blob: bytes,
    mesh_index: int,
    primitive_index: int,
) -> Mesh:
    """Build a UV-textured Mesh from a GLB primitive.

    glTF shares one index buffer per vertex, so the primitive indices are both
    the geometry faces and the raw UV faces; the base-color image becomes the
    UV texture map on convention ``"top_left"``.

    Args:
        gltf: Decoded glTF JSON dictionary.
        binary_blob: GLB BIN chunk bytes.
        mesh_index: Index into ``gltf["meshes"]``.
        primitive_index: Index into the mesh's ``primitives``.

    Returns:
        One `Mesh` carrying a `MeshTextureUVTextureMap` on convention ``"top_left"``.
    """

    primitive = gltf["meshes"][mesh_index]["primitives"][primitive_index]
    verts_np = read_accessor(
        gltf=gltf,
        binary_blob=binary_blob,
        accessor_index=primitive["attributes"]["POSITION"],
    )
    indices_np = read_accessor(
        gltf=gltf, binary_blob=binary_blob, accessor_index=primitive["indices"]
    ).reshape(-1, 3)
    verts_uvs_np = read_accessor(
        gltf=gltf,
        binary_blob=binary_blob,
        accessor_index=primitive["attributes"]["TEXCOORD_0"],
    )
    image_index = _resolve_base_color_texture_image_index(
        gltf=gltf, primitive=primitive
    )
    image_bytes = read_image_bytes(
        gltf=gltf, binary_blob=binary_blob, image_index=image_index
    )
    uv_texture_map = (
        decode_image_bytes(image_bytes=image_bytes).to(torch.float32) / 255.0
    )

    verts_uvs = torch.from_numpy(verts_uvs_np.astype(np.float32)).contiguous()
    faces_uvs = torch.from_numpy(indices_np.astype(np.int64)).contiguous()
    canonical_verts_uvs, canonical_faces_uvs = shift_seam_crossing_faces_to_seam_safe(
        verts_uvs=verts_uvs,
        faces_uvs=faces_uvs,
    )
    return Mesh(
        verts=torch.from_numpy(verts_np.astype(np.float32)).contiguous(),
        faces=torch.from_numpy(indices_np.astype(np.int64)).contiguous(),
        texture=MeshTextureUVTextureMap(
            uv_texture_map=uv_texture_map.contiguous(),
            verts_uvs=canonical_verts_uvs.contiguous(),
            faces_uvs=canonical_faces_uvs.contiguous(),
            convention="top_left",
        ),
    )


def _select_mesh_primitive(gltf: Dict[str, Any]) -> Tuple[int, int]:
    """Select the (mesh_index, primitive_index) to load.

    Prefers the primitive whose material carries a base-color texture (for a GLB
    of many untextured marker meshes plus one textured face, this uniquely picks
    the face); otherwise falls back to the primitive with the most vertices.

    Args:
        gltf: Decoded glTF JSON dictionary.

    Returns:
        Tuple ``(mesh_index, primitive_index)``.
    """

    assert gltf.get(
        "meshes"
    ), f"Expected the GLB to contain at least one mesh, got keys={list(gltf.keys())}"
    materials = gltf.get("materials", [])
    textured_choice: Tuple[int, int] = None
    largest_choice: Tuple[int, int] = None
    largest_vertex_count = -1
    for mesh_index, mesh in enumerate(gltf["meshes"]):
        for primitive_index, primitive in enumerate(mesh.get("primitives", [])):
            material_index = primitive.get("material")
            is_textured = (
                material_index is not None
                and material_index < len(materials)
                and "baseColorTexture"
                in materials[material_index].get("pbrMetallicRoughness", {})
            )
            position_accessor = primitive.get("attributes", {}).get("POSITION")
            vertex_count = (
                int(gltf["accessors"][position_accessor]["count"])
                if position_accessor is not None
                else 0
            )
            if is_textured and textured_choice is None:
                textured_choice = (mesh_index, primitive_index)
            if vertex_count > largest_vertex_count:
                largest_vertex_count = vertex_count
                largest_choice = (mesh_index, primitive_index)
    assert (
        largest_choice is not None
    ), f"Expected the GLB to contain at least one mesh primitive, got meshes={len(gltf['meshes'])}"
    return textured_choice if textured_choice is not None else largest_choice


def _resolve_base_color_texture_image_index(
    gltf: Dict[str, Any],
    primitive: Dict[str, Any],
) -> int:
    """Resolve a primitive's material base-color texture to its glTF image index.

    Args:
        gltf: Decoded glTF JSON dictionary.
        primitive: The selected mesh primitive dict.

    Returns:
        Image index into ``gltf["images"]``.
    """

    material_index = primitive["material"]
    base_color_texture = gltf["materials"][material_index]["pbrMetallicRoughness"][
        "baseColorTexture"
    ]
    texture_index = base_color_texture["index"]
    image_index = gltf["textures"][texture_index]["source"]
    return int(image_index)
