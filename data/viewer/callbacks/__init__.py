"""Initialize all viewer callbacks.

This module imports all callback modules to ensure they are registered with the callback registry.
"""
from data.viewer.callbacks.registry import registry, callback
from data.viewer.callbacks import dataset
from data.viewer.callbacks import navigation
from data.viewer.callbacks import three_d_settings
from data.viewer.callbacks import camera
from data.viewer.callbacks import transforms
from data.viewer.callbacks import backend_sync
from data.viewer.callbacks import class_distribution
from data.viewer.callbacks import segmentation_color


__all__ = (
    'registry',
    'callback',
    'dataset',
    'navigation',
    'three_d_settings',
    'camera',
    'transforms',
    'backend_sync',
    'class_distribution',
    'segmentation_color',
)
