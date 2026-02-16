from typing import TYPE_CHECKING, Any, Optional

import dash
from dash.dependencies import Input, Output, State

if TYPE_CHECKING:
    from data.viewer.three_d_scene.three_d_scene_viewer import ThreeDSceneViewer


def register_translation_slider_callbacks(
    app: dash.Dash, viewer: "ThreeDSceneViewer"
) -> None:
    """Update translation step display from slider."""
    # Input validations
    assert isinstance(app, dash.Dash), f"app must be Dash, got {type(app)}"
    assert viewer is not None, "viewer must not be None"

    @app.callback(
        Output("translation-step-display", "children", allow_duplicate=True),
        Input("translation-step-slider", "value"),
        State("dataset-dropdown", "value"),
        State("scene-dropdown", "value"),
        prevent_initial_call=True,
    )
    def update_translation_step_display(
        slider_value: Optional[float],
        _dataset_value: Optional[str],
        _scene_value: Optional[str],
    ) -> str:
        # Input validations
        assert slider_value is None or isinstance(
            slider_value, (float, int)
        ), "slider_value must be float, int, or None"
        assert _dataset_value is None or isinstance(
            _dataset_value, str
        ), "_dataset_value must be str or None"
        assert _scene_value is None or isinstance(
            _scene_value, str
        ), "_scene_value must be str or None"

        if viewer.current_dataset is None or viewer.current_scene is None:
            return "Step: --"
        normalized = 0.1 if slider_value is None else float(slider_value)
        step_value = viewer.get_translation_step(normalized)
        return f"Step: {step_value:.2f}"
