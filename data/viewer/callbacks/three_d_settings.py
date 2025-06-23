"""3D settings-related callbacks for the viewer."""
from typing import Dict, List, Optional, Union
from dash import Input, Output, html
from dash.exceptions import PreventUpdate
from data.viewer.callbacks.registry import callback, registry


# Dataset types that use 3D visualization
THREE_D_DATASET_TYPES = ['3dcd', 'pcr']

@callback(
    outputs=[
        Output('3d-settings-store', 'data')
    ],
    inputs=[
        Input('point-size-slider', 'value'),
        Input('point-opacity-slider', 'value'),
        Input('radius-slider', 'value'),
        Input('correspondence-radius-slider', 'value')
    ],
    group="3d_settings"
)
def update_3d_settings(
    point_size: float,
    point_opacity: float,
    radius: float,
    correspondence_radius: float
) -> List[Dict[str, Union[str, int, float, bool]]]:
    """Update 3D settings store when slider values change."""
    if point_size is None or point_opacity is None:
        raise PreventUpdate

    # Update backend state with new 3D settings
    registry.viewer.backend.update_state(
        point_size=point_size,
        point_opacity=point_opacity,
        radius=radius or 0.05,
        correspondence_radius=correspondence_radius or 0.1
    )

    # Store all 3D settings in the store
    settings = {
        'point_size': point_size,
        'point_opacity': point_opacity,
        'radius': radius or 0.05,  # Default radius
        'correspondence_radius': correspondence_radius or 0.1  # Default correspondence radius
    }

    return [settings]


@callback(
    outputs=[
        Output('view-controls', 'style'),
        Output('pcr-controls', 'style')
    ],
    inputs=[Input('dataset-info', 'data')],
    group="display",
)
def update_view_controls(
    dataset_info: Optional[Dict[str, Union[str, int, bool, Dict]]]
) -> List[Dict[str, str]]:
    """Update the visibility of 3D view controls based on dataset type."""
    if dataset_info is None or not dataset_info:
        return [{'display': 'none'}, {'display': 'none'}]

    assert 'type' in dataset_info, f"{dataset_info.keys()=}"
    dataset_type = dataset_info.get('type')

    # Default styles
    view_controls_style = {'display': 'none'}
    pcr_controls_style = {'display': 'none'}

    # Show 3D controls for 3D datasets
    if dataset_type in THREE_D_DATASET_TYPES:
        view_controls_style = {'display': 'block'}

        # Show PCR controls only for PCR datasets
        if dataset_type == 'pcr':
            pcr_controls_style = {'display': 'block'}

    return [view_controls_style, pcr_controls_style]
