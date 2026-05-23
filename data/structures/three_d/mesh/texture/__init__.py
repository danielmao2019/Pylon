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
from data.structures.three_d.mesh.texture.validate_uv_texture_map import (
    validate_uv_texture_map,
)
from data.structures.three_d.mesh.texture.validate_vertex_color import (
    validate_vertex_color,
)

__all__ = (
    "MeshTexture",
    "MeshTextureVertexColor",
    "MeshTextureUVTextureMap",
    "transform_verts_uvs_convention",
    "validate_vertex_color",
    "validate_uv_texture_map",
)
