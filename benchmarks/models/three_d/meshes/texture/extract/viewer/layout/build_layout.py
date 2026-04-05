"""Build the layout for the texture-extraction benchmark viewer."""

from dash import Dash

from benchmarks.models.three_d.meshes.texture.extract.viewer.layout.components import (
    build_root_layout,
    build_scene_panel_children,
)
from benchmarks.models.three_d.meshes.texture.extract.viewer.viewer_state import (
    get_default_scene_name,
    get_scene_names,
    get_scene_payload,
)


def build_layout(
    app: Dash,
) -> None:
    """Build and assign the benchmark viewer layout.

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

    default_scene_name = get_default_scene_name()
    initial_scene_payload = get_scene_payload(scene_name=default_scene_name)
    app.layout = build_root_layout(
        scene_names=get_scene_names(),
        default_scene_name=default_scene_name,
        right_panel_children=build_scene_panel_children(
            scene_payload=initial_scene_payload,
        ),
    )
