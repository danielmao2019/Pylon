"""
DATA.TRANSFORMS.RESIZE API
"""
from data.transforms.resize.bboxes import ResizeBBoxes
from data.transforms.resize.normals import ResizeNormals
from data.transforms.resize.maps import ResizeMaps


__all__ = (
    'ResizeBBoxes',
    'ResizeNormals',
    'ResizeMaps',
)
