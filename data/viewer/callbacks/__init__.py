"""Initialize all viewer callbacks.

This module imports all callback modules to ensure they are registered with the callback registry.
"""
from data.viewer.callbacks.registry import registry, callback
from data.viewer.callbacks import dataset
from data.viewer.callbacks import navigation
from data.viewer.callbacks import display
from data.viewer.callbacks import three_d_settings


__all__ = (
    'registry',
    'callback',
    'dataset',
    'navigation',
    'display',
    'three_d_settings',
)
