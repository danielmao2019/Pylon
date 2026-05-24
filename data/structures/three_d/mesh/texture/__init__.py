"""
DATA.STRUCTURES.THREE_D.MESH.TEXTURE API
"""

from data.structures.three_d.mesh.texture.conventions import (
    transform_verts_uvs_convention,
)
from data.structures.three_d.mesh.texture.mesh_texture import MeshTexture
from data.structures.three_d.mesh.texture.mesh_texture_uv_texture_map import (
    MeshTextureUVTextureMap,
)
from data.structures.three_d.mesh.texture.mesh_texture_vertex_color import (
    MeshTextureVertexColor,
)
from data.structures.three_d.mesh.texture.texel_face_map import (
    build_texel_face_map,
)
from data.structures.three_d.mesh.texture.validate_uv_texture_map import (
    validate_faces_uvs,
    validate_mesh_uv_convention,
    validate_uv_texture_map,
    validate_uv_texture_map_image,
    validate_verts_uvs,
)
from data.structures.three_d.mesh.texture.validate_vertex_color import (
    validate_vertex_color,
)

__all__ = (
    "MeshTexture",
    "MeshTextureVertexColor",
    "MeshTextureUVTextureMap",
    "build_texel_face_map",
    "transform_verts_uvs_convention",
    "validate_faces_uvs",
    "validate_mesh_uv_convention",
    "validate_uv_texture_map",
    "validate_uv_texture_map_image",
    "validate_verts_uvs",
    "validate_vertex_color",
)
