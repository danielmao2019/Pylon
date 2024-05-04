"""
UTILS.INPUT_CHECKS API
"""
# str types
from utils.input_checks.str_types import check_read_file
from utils.input_checks.str_types import check_write_file
from utils.input_checks.str_types import check_read_dir
from utils.input_checks.str_types import check_write_dir
# tensor types
from utils.input_checks.tensor_types import check_semantic_segmentation, check_semantic_segmentation_pred, check_semantic_segmentation_true
from utils.input_checks.tensor_types import check_depth_estimation
from utils.input_checks.tensor_types import check_normal_estimation


__all__ = (
    # str types
    'check_read_file',
    'check_write_file',
    'check_read_dir',
    'check_write_dir',
    # tensor types
    'check_semantic_segmentation',
    'check_semantic_segmentation_pred',
    'check_semantic_segmentation_true',
    'check_depth_estimation',
    'check_normal_estimation',
)
