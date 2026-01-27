"""Dataset-related callbacks for the viewer."""

from typing import TYPE_CHECKING, Any, Dict, List, Optional

import dash
from dash import html
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate

from data.viewer.layout.controls.transforms import create_transforms_section

if TYPE_CHECKING:
    from data.viewer.viewer import DatasetViewer


# =========================================================================
# Callback Registration
# =========================================================================


def register_transforms_section_callbacks(
    app: dash.Dash, viewer: "DatasetViewer"
) -> None:
    # Input validations
    assert isinstance(app, dash.Dash), f"app must be Dash, got {type(app)}"
    assert hasattr(viewer, "backend"), f"viewer must expose backend, got {type(viewer)}"

    # =====================================================================
    # Dataset Loading and UI Management
    # =====================================================================

    @app.callback(
        [Output('transforms-section', 'children', allow_duplicate=True)],
        Input('dataset-info', 'data'),
    )
    def update_transforms_section(
        dataset_info: Optional[Dict[str, Any]],
    ) -> List[html.Div]:
        """Update the transforms section when dataset info changes."""
        # Input validations
        assert dataset_info is None or isinstance(
            dataset_info, dict
        ), f"dataset_info must be dict or None, got {type(dataset_info)}"
        assert (
            dataset_info is None or dataset_info == {} or ('transforms' in dataset_info)
        ), "dataset_info must include transforms"

        if not dataset_info:
            raise PreventUpdate

        transforms = dataset_info['transforms']
        transforms_section = create_transforms_section(transforms)

        return [transforms_section]
