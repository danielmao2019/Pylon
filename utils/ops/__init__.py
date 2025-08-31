"""
UTILS.OPS API
"""
from utils.ops.apply import apply_tensor_op, apply_pairwise
from utils.ops.dict_as_tensor import (
    buffer_equal, buffer_allclose,
    buffer_add, buffer_scalar_mul, buffer_sub,
    buffer_mul, buffer_rec, buffer_div,
    buffer_mean,
    transpose_buffer,
)
from utils.ops.materialize_tensor import materialize_tensor


__all__ = (
    'apply_tensor_op',
    'apply_pairwise',
    'buffer_equal',
    'buffer_allclose',
    'buffer_add',
    'buffer_scalar_mul',
    'buffer_sub',
    'buffer_mul',
    'buffer_rec',
    'buffer_div',
    'buffer_mean',
    'transpose_buffer',
    'materialize_tensor',
)
