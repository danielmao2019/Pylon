"""Callback handlers for the dataset viewer.

This module contains all the callback handlers for the dataset viewer application.
The callbacks handle user interactions and update the UI accordingly.
"""

from data.viewer.callbacks.dataset import register_dataset_callbacks
from data.viewer.callbacks.display import register_display_callbacks
from data.viewer.callbacks.navigation import register_navigation_callbacks
from data.viewer.callbacks.transforms import register_transform_callbacks

__all__ = [
    'register_dataset_callbacks',
    'register_display_callbacks',
    'register_navigation_callbacks',
    'register_transform_callbacks'
] 