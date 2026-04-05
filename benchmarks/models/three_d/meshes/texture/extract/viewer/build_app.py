"""Dash app factory for the texture-extraction benchmark viewer."""

from pathlib import Path

from dash import Dash

from benchmarks.models.three_d.meshes.texture.extract.viewer.callbacks.register import (
    register_callbacks,
)
from benchmarks.models.three_d.meshes.texture.extract.viewer.layout.build_layout import (
    build_layout,
)
from benchmarks.models.three_d.meshes.texture.extract.viewer.viewer_state import (
    configure_viewer,
)


def build_app(
    results_root: Path,
) -> Dash:
    """Build the texture-extraction benchmark Dash app.

    Args:
        results_root: Benchmark results root.

    Returns:
        Configured Dash app.
    """

    def _validate_inputs() -> None:
        """Validate input arguments.

        Args:
            None.

        Returns:
            None.
        """

        assert isinstance(results_root, Path), (
            "Expected `results_root` to be a `Path`. " f"{type(results_root)=}."
        )
        assert results_root.exists(), (
            "Expected `results_root` to exist. " f"{results_root=}"
        )
        assert results_root.is_dir(), (
            "Expected `results_root` to be a directory. " f"{results_root=}"
        )

    _validate_inputs()

    configure_viewer(results_root=results_root)
    app = Dash(__name__, title="Texture Extraction Benchmark")
    build_layout(app=app)
    register_callbacks(app=app)
    return app
