"""
UTILS.OPS API
"""

from utils.ops.apply import apply_pairwise, apply_tensor_op
from utils.ops.chunked_matmul import chunked_matmul
from utils.ops.dict_as_tensor import (
    buffer_add,
    buffer_allclose,
    buffer_div,
    buffer_equal,
    buffer_mean,
    buffer_mul,
    buffer_rec,
    buffer_scalar_mul,
    buffer_sub,
    transpose_buffer,
)
from utils.ops.materialize_tensor import materialize_tensor

__all__ = (
    'apply_tensor_op',
    'apply_pairwise',
    'chunked_matmul',
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
