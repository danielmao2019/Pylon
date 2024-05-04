"""
DATASETS.TRANSFORMS.RESIZE API
"""
from datasets.transforms.resize.bboxes import ResizeBBoxes
from datasets.transforms.resize.normals import ResizeNormals
from datasets.transforms.resize.maps import ResizeMaps


__all__ = (
    'ResizeBBoxes',
    'ResizeNormals',
    'ResizeMaps',
)
