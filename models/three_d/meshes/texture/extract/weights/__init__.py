"""
MODELS.THREE_D.MESHES.TEXTURE.EXTRACT.WEIGHTS API
"""

from models.three_d.meshes.texture.extract.weights.normal_weights import (
    _compute_f_normals_weights,
    _compute_v_normals_weights,
)
from models.three_d.meshes.texture.extract.weights.weights_cfg import (
    WEIGHTS_CFG_ALLOWED_KEYS,
    normalize_weights_cfg,
    validate_weights_cfg,
)

__all__ = (
    "WEIGHTS_CFG_ALLOWED_KEYS",
    "_compute_f_normals_weights",
    "_compute_v_normals_weights",
    "normalize_weights_cfg",
    "validate_weights_cfg",
)
