"""
UTILS.MODELS API
"""

from utils.models.grads import get_flattened_grads
from utils.models.params import get_flattened_params

__all__ = (
    'get_flattened_params',
    'get_flattened_grads',
)
