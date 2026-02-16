"""3D settings-related callbacks for the viewer."""

from typing import TYPE_CHECKING, Dict, List, Optional, Union

import dash
from dash.dependencies import Input, Output

if TYPE_CHECKING:
    from data.viewer.dataset.viewer import DatasetViewer


def register_three_d_settings_view_controls_callbacks(
    app: dash.Dash, viewer: "DatasetViewer"
) -> None:
    # Input validations
    assert isinstance(app, dash.Dash), f"app must be Dash, got {type(app)}"
    assert hasattr(viewer, "backend"), f"viewer must expose backend, got {type(viewer)}"

    @app.callback(
        [Output('view-controls', 'style'), Output('pcr-controls', 'style')],
        [Input('dataset-info', 'data')],
    )
    def update_view_controls(
        dataset_info: Optional[Dict[str, Union[str, int, bool, Dict]]],
    ) -> List[Dict[str, str]]:
        """Update the visibility of 3D view controls based on dataset type."""
        # Input validations
        assert dataset_info is None or isinstance(
            dataset_info, dict
        ), f"dataset_info must be dict or None, got {type(dataset_info)}"
        assert (
            dataset_info is None
            or dataset_info == {}
            or ('requires_3d_visualization' in dataset_info)
        ), "dataset_info must include requires_3d_visualization"
        assert (
            dataset_info is None or dataset_info == {} or ('type' in dataset_info)
        ), "dataset_info must include type"

        if dataset_info is None or dataset_info == {}:
            return [{'display': 'none'}, {'display': 'none'}]

        requires_3d = dataset_info['requires_3d_visualization']
        dataset_type = dataset_info['type']

        # Default styles
        view_controls_style = {'display': 'none'}
        pcr_controls_style = {'display': 'none'}

        # Show 3D controls for 3D datasets
        if requires_3d:
            view_controls_style = {'display': 'block'}

            # Show PCR controls only for PCR datasets
            if dataset_type == 'pcr':
                pcr_controls_style = {'display': 'block'}

        return [view_controls_style, pcr_controls_style]
