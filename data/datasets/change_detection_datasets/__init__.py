"""Change Detection datasets module.

This module contains dataset classes for both 2D and 3D change detection,
including base classes with built-in display functionality.
"""
from .base_2d_cd_dataset import Base2DCDDataset
from .base_3d_cd_dataset import Base3DCDDataset

__all__ = [
    'Base2DCDDataset',
    'Base3DCDDataset',
]
