"""
UTILS.OPS API
"""
from utils.ops.apply import apply_tensor_op
from utils.ops.apply import apply_pairwise
from utils.ops.dict_as_tensor import buffer_equal
from utils.ops.dict_as_tensor import transpose_buffer
from utils.ops.dict_as_tensor import average_buffer


__all__ = (
    'apply_tensor_op',
    'apply_pairwise',
    'buffer_equal',
    'transpose_buffer',
    'average_buffer',
)
