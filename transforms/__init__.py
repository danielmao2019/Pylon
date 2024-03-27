"""
TRANSFORMS API.
"""
from transforms.normalize_image import NormalizeImage
from transforms.normalize_depth import NormalizeDepth
from transforms.resize_bboxes import ResizeBBoxes
from transforms.resize_normal_estimation import ResizeNormalEstimation


__all__ = (
    'NormalizeImage',
    'NormalizeDepth',
    'ResizeBBoxes',
    'ResizeNormalEstimation',
)
