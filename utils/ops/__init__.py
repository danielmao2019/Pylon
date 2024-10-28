"""
UTILS.OPS API
"""
from utils.ops.apply import apply_tensor_op, apply_pairwise
from utils.ops.dict_as_tensor import buffer_equal, buffer_add, buffer_scalar_mul, buffer_mean
from utils.ops.dict_as_tensor import transpose_buffer


__all__ = (
    'apply_tensor_op',
    'apply_pairwise',
    'buffer_equal',
    'buffer_add',
    'buffer_scalar_mul',
    'buffer_mean',
    'transpose_buffer',
)
