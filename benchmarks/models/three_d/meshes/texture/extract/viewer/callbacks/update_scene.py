"""Scene-selection callback for the texture-extraction benchmark viewer."""

from typing import Any, List

from dash import Dash, Input, Output, ctx
from dash.exceptions import PreventUpdate

from benchmarks.models.three_d.meshes.texture.extract.viewer.layout.components import (
    RIGHT_PANEL_ID,
    SCENE_RADIO_ID,
    build_scene_panel_children,
)
from benchmarks.models.three_d.meshes.texture.extract.viewer.viewer_state import (
    get_scene_payload,
)


def register_scene_callback(
    app: Dash,
) -> None:
    """Register the scene-selection callback.

    Args:
        app: Dash app instance.

    Returns:
        None.
    """

    def _validate_inputs() -> None:
        """Validate input arguments.

        Args:
            None.

        Returns:
            None.
        """

        assert isinstance(app, Dash), (
            "Expected `app` to be a Dash instance. " f"{type(app)=}."
        )

    _validate_inputs()

    def validate_scene_trigger(
        scene_name: str,
    ) -> None:
        """Validate the scene-selection callback trigger.

        Args:
            scene_name: Selected scene name.

        Returns:
            None.
        """

        assert isinstance(scene_name, str), (
            "Expected `scene_name` to be a string. " f"{type(scene_name)=}."
        )
        if ctx.triggered_id != SCENE_RADIO_ID:
            raise PreventUpdate

    @app.callback(
        Output(RIGHT_PANEL_ID, "children"),
        Input(SCENE_RADIO_ID, "value"),
    )
    def update_scene_panel(
        scene_name: str,
    ) -> List[Any]:
        """Update the right panel when the scene selection changes.

        Args:
            scene_name: Selected scene name.

        Returns:
            Right-panel child component list.
        """

        validate_scene_trigger(scene_name=scene_name)

        scene_payload = get_scene_payload(scene_name=scene_name)
        return build_scene_panel_children(scene_payload=scene_payload)
