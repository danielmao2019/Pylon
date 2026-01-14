"""Callback registration modules for iVISION 4D Scene Viewer.

Each module contains functional callbacks for specific modalities or features.
"""

from data.viewer.ivision.callbacks.camera_selector import (
    register_camera_selector_callbacks,
)
from data.viewer.ivision.callbacks.dataset_selection import (
    register_dataset_selection_callbacks,
)
from data.viewer.ivision.callbacks.keyboard import register_keyboard_callbacks
from data.viewer.ivision.callbacks.recording import register_recording_callbacks
from data.viewer.ivision.callbacks.rotation_slider import (
    register_rotation_slider_callbacks,
)
from data.viewer.ivision.callbacks.scene_selection import (
    register_scene_selection_callbacks,
)
from data.viewer.ivision.callbacks.translation_slider import (
    register_translation_slider_callbacks,
)
from data.viewer.ivision.callbacks.camera_overlay_toggle import (
    register_camera_overlay_toggle_callbacks,
)


def register_viewer_callbacks(app, viewer) -> None:
    register_dataset_selection_callbacks(app=app, viewer=viewer)
    register_scene_selection_callbacks(app=app, viewer=viewer)
    register_camera_selector_callbacks(app=app, viewer=viewer)
    register_keyboard_callbacks(app=app, viewer=viewer)
    register_translation_slider_callbacks(app=app, viewer=viewer)
    register_rotation_slider_callbacks(app=app, viewer=viewer)
    register_recording_callbacks(app=app, viewer=viewer)
    register_camera_overlay_toggle_callbacks(app=app, viewer=viewer)
