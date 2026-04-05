"""
MODELS.THREE_D.MESHES.TEXTURE.EXTRACT.VISIBILITY API
"""

from models.three_d.meshes.texture.extract.visibility.texel_visibility import (
    compute_f_visibility_mask,
)
from models.three_d.meshes.texture.extract.visibility.texel_visibility_v2 import (
    compute_f_visibility_mask_v2,
)
from models.three_d.meshes.texture.extract.visibility.vertex_visibility import (
    compute_v_visibility_mask,
)

__all__ = (
    "compute_f_visibility_mask",
    "compute_f_visibility_mask_v2",
    "compute_v_visibility_mask",
)
