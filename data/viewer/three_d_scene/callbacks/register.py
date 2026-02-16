"""Register callbacks for the 3D scene viewer."""

from typing import TYPE_CHECKING

import dash

from data.viewer.three_d_scene.callbacks.camera_overlay_toggle import (
    register_camera_overlay_toggle_callbacks,
)
from data.viewer.three_d_scene.callbacks.camera_selector import (
    register_camera_selector_callbacks,
)
from data.viewer.three_d_scene.callbacks.dataset_selection import (
    register_dataset_selection_callbacks,
)
from data.viewer.three_d_scene.callbacks.keyboard import register_keyboard_callbacks
from data.viewer.three_d_scene.callbacks.recording import register_recording_callbacks
from data.viewer.three_d_scene.callbacks.rotation_slider import (
    register_rotation_slider_callbacks,
)
from data.viewer.three_d_scene.callbacks.scene_selection import (
    register_scene_selection_callbacks,
)
from data.viewer.three_d_scene.callbacks.translation_slider import (
    register_translation_slider_callbacks,
)

if TYPE_CHECKING:
    from data.viewer.three_d_scene.three_d_scene_viewer import ThreeDSceneViewer


def register_viewer_callbacks(
    app: dash.Dash, viewer: "ThreeDSceneViewer"
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
