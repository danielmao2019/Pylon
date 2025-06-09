"""3D settings-related callbacks for the viewer."""
from typing import Dict, List, Optional, Union
from dash import Input, Output, html
from dash.exceptions import PreventUpdate
from data.viewer.callbacks.registry import callback, registry


# Dataset types that use 3D visualization
THREE_D_DATASET_TYPES = ['3dcd', 'pcr']

@callback(
    outputs=[
        Output('3d-settings-section', 'children'),
        Output('3d-settings-store', 'data')
    ],
    inputs=[
        Input('3d-settings-dropdown', 'value'),
        Input('3d-settings-params', 'data')
    ],
    group="3d_settings"
)
def update_3d_settings(
    selected_setting: Optional[str],
    setting_params: Optional[Dict[str, Union[str, int, float, bool]]]
) -> List[Union[html.Div, Dict[str, Union[str, int, float, bool]]]]:
    """Update the 3D settings section when a setting is selected or parameters change."""
    if selected_setting is None and setting_params is None:
        raise PreventUpdate

    # Get current dataset info
    dataset_info: Dict[str, Union[str, int, bool, Dict]] = registry.viewer.state.get_state()['dataset_info']
    if not dataset_info:
        return [
            html.Div("No dataset loaded."),
            registry.viewer.state.get_state()['3d_settings']
        ]
        
    assert 'type' in dataset_info, f"{dataset_info.keys()=}"
    dataset_type = dataset_info.get('type')

    if dataset_type not in THREE_D_DATASET_TYPES:
        return [
            html.Div("3D settings are only available for 3D datasets."),
            registry.viewer.state.get_state()['3d_settings']
        ]

    # Update state with new 3D settings
    registry.viewer.state.update_3d_settings(selected_setting, setting_params)

    # Create updated 3D settings section
    settings_section: html.Div = create_3d_settings_section()

    return [
        settings_section,
        registry.viewer.state.get_state()['3d_settings']
    ]


@callback(
    outputs=[
        Output('view-controls', 'style'),
        Output('pcr-controls', 'style')
    ],
    inputs=[Input('dataset-info', 'data')],
    group="display"
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
