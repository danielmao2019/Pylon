"""
DATA.STRUCTURES.THREE_D.MESH API
"""

from data.structures.three_d.mesh.convert import (
    mesh_from_open3d,
    mesh_from_pytorch3d,
    mesh_from_trimesh,
    mesh_to_open3d,
    mesh_to_pytorch3d,
    mesh_to_trimesh,
)
from data.structures.three_d.mesh.load import load_mesh
from data.structures.three_d.mesh.merge import merge_meshes
from data.structures.three_d.mesh.mesh import Mesh
from data.structures.three_d.mesh.save import save_mesh
from data.structures.three_d.mesh.texture import (
    MeshTexture,
    MeshTextureUVTextureMap,
    MeshTextureVertexColor,
    transform_vertex_uv_convention,
    validate_uv_texture_map,
    validate_vertex_color,
)
from data.structures.three_d.mesh.validate import (
    validate_faces,
    validate_mesh_attributes,
    validate_vertices,
)

__all__ = (
    "Mesh",
    "load_mesh",
    "save_mesh",
    "merge_meshes",
    "MeshTexture",
    "MeshTextureVertexColor",
    "MeshTextureUVTextureMap",
    "transform_vertex_uv_convention",
    "mesh_from_open3d",
    "mesh_from_pytorch3d",
    "mesh_from_trimesh",
    "mesh_to_open3d",
    "mesh_to_pytorch3d",
    "mesh_to_trimesh",
    "validate_vertices",
    "validate_faces",
    "validate_vertex_color",
    "validate_uv_texture_map",
    "validate_mesh_attributes",
)
