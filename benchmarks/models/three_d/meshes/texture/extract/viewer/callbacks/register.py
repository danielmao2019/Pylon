"""Aggregate callback registration for the texture-extraction benchmark viewer."""

from dash import Dash

from benchmarks.models.three_d.meshes.texture.extract.viewer.callbacks.update_scene import (
    register_scene_callback,
)


def register_callbacks(
    app: Dash,
) -> None:
    """Register all benchmark viewer callbacks.

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

    register_scene_callback(app=app)
