"""Callback handlers for the dataset viewer.

This module contains all the callback handlers for the dataset viewer application.
The callbacks handle user interactions and update the UI accordingly.
"""

from data.viewer.callbacks.registry import registry
from data.viewer.callbacks import dataset, display, settings_3d, transforms, navigation, index  # import all callback modules

__all__ = [
    'register_all_callbacks'
]

def register_all_callbacks(app, viewer):
    """Register all callbacks with the Dash app using the registry system."""
    # All callbacks are automatically registered via the @callback decorator
    # Just need to register them with the app
    registry.register_callbacks(app)
