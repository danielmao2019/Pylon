"""Register callbacks for the iVISION 4D Scene Viewer."""

from typing import TYPE_CHECKING

import dash

from data.viewer.ivision.callbacks.camera_overlay_toggle import (
    register_camera_overlay_toggle_callbacks,
)
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

if TYPE_CHECKING:
    from data.viewer.ivision.ivision_4d_scene_viewer import iVISION_4D_Scene_Viewer


def register_viewer_callbacks(
    app: dash.Dash, viewer: "iVISION_4D_Scene_Viewer"
) -> None:
    # Input validations
    assert isinstance(app, dash.Dash), f"app must be Dash, got {type(app)}"
    assert viewer is not None, "viewer must not be None"

    register_dataset_selection_callbacks(app=app, viewer=viewer)
    register_scene_selection_callbacks(app=app, viewer=viewer)
    register_camera_selector_callbacks(app=app, viewer=viewer)
    register_keyboard_callbacks(app=app, viewer=viewer)
    register_translation_slider_callbacks(app=app, viewer=viewer)
    register_rotation_slider_callbacks(app=app, viewer=viewer)
    register_recording_callbacks(app=app, viewer=viewer)
    register_camera_overlay_toggle_callbacks(app=app, viewer=viewer)
