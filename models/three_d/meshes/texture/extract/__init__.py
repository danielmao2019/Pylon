"""
MODELS.THREE_D.MESHES.TEXTURE.EXTRACT API
"""

from models.three_d.meshes.texture.extract.camera_geometry import (
    _camera_vertices_to_clip,
    _render_camera_depth_buffer,
    _render_camera_face_index_buffer,
    _vertices_world_to_camera,
)
from models.three_d.meshes.texture.extract.extract import (
    _build_camera_uv_interpolation_data,
    _build_uv_rasterization_data,
    _build_uv_rasterization_mesh,
    _compute_f_normals_weights,
    _extract_uv_texture_map_from_single_image,
    _extract_vertex_color_from_single_image,
    extract_texture_from_images,
)
from models.three_d.meshes.texture.extract.visibility import (
    compute_f_visibility_mask,
    compute_f_visibility_mask_v2,
    compute_v_visibility_mask,
)

__all__ = (
    "_build_camera_uv_interpolation_data",
    "_build_uv_rasterization_data",
    "_build_uv_rasterization_mesh",
    "_camera_vertices_to_clip",
    "_compute_f_normals_weights",
    "_extract_uv_texture_map_from_single_image",
    "_extract_vertex_color_from_single_image",
    "_render_camera_depth_buffer",
    "_render_camera_face_index_buffer",
    "_vertices_world_to_camera",
    "compute_f_visibility_mask",
    "compute_f_visibility_mask_v2",
    "compute_v_visibility_mask",
    "extract_texture_from_images",
)
