from typing import TYPE_CHECKING, Any, Optional

import dash
from dash.dependencies import Input, Output, State

if TYPE_CHECKING:
    from data.viewer.ivision.ivision_4d_scene_viewer import iVISION_4D_Scene_Viewer


def register_translation_slider_callbacks(
    app: dash.Dash, viewer: "iVISION_4D_Scene_Viewer"
) -> None:
    """Update translation step display from slider."""

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
        if viewer.current_dataset is None or viewer.current_scene is None:
            return "Step: --"
        normalized = 0.1 if slider_value is None else float(slider_value)
        step_value = viewer.get_translation_step(normalized)
        return f"Step: {step_value:.2f}"
