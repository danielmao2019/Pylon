"""
MODELS.THREE_D.MESHES.RENDER API
"""

from models.three_d.meshes.render.core import render_rgb_from_mesh
from models.three_d.meshes.render.display import render_display
from models.three_d.meshes.render.shading import compute_sh_shading
from models.three_d.meshes.render.uv_texture import render_uv_texture_aligned

__all__ = (
    "render_rgb_from_mesh",
    "render_display",
    "compute_sh_shading",
    "render_uv_texture_aligned",
)
