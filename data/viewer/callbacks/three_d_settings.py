"""3D settings-related callbacks for the viewer."""
from typing import Dict, List, Optional, Union
from dash import Input, Output
from dash.exceptions import PreventUpdate
from data.viewer.callbacks.registry import callback
from data.viewer.utils.settings_config import ViewerSettings


@callback(
    outputs=[
        Output('3d-settings-store', 'data')
    ],
    inputs=[
        Input('point-size-slider', 'value'),
        Input('point-opacity-slider', 'value'),
        Input('radius-slider', 'value'),
        Input('correspondence-radius-slider', 'value'),
        Input('lod-type-dropdown', 'value')
    ],
    group="3d_settings"
)
def update_3d_settings(
    point_size: float,
    point_opacity: float,
    sym_diff_radius: float,
    corr_radius: float,
    lod_type: str
) -> List[Dict[str, Union[str, int, float, bool]]]:
    """Update 3D settings store when slider values change.
    
    This is now a PURE UI callback - it only updates the UI store.
    Backend synchronization happens in backend_sync.py automatically.
    
    Args:
        point_size: Size of points in the 3D visualization
        point_opacity: Opacity of points (0.0 to 1.0)
        sym_diff_radius: Radius for symmetric difference computation
        corr_radius: Radius for correspondence visualization
        lod_type: Type of LOD to use ("continuous", "discrete", or "none")
    
    Returns:
        List containing a dictionary of all 3D settings for UI store
    """
    if point_size is None or point_opacity is None:
        raise PreventUpdate

    # Create settings dictionary with provided values
    raw_settings = {
        'point_size': point_size,
        'point_opacity': point_opacity,
        'sym_diff_radius': sym_diff_radius,
        'corr_radius': corr_radius,
        'lod_type': lod_type
    }
    
    # Validate and apply defaults using centralized configuration
    settings = ViewerSettings.validate_3d_settings(
        ViewerSettings.get_3d_settings_with_defaults(raw_settings)
    )

    # PURE UI PATTERN: Only return UI store data
    # Backend sync happens automatically in backend_sync.py
    return [settings]


@callback(
    outputs=[
        Output('view-controls', 'style'),
        Output('pcr-controls', 'style')
    ],
    inputs=[Input('dataset-info', 'data')],
    group="3d_settings",
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

    # Show 3D controls for 3D datasets using proper API
    if ViewerSettings.requires_3d_visualization(dataset_type):
        view_controls_style = {'display': 'block'}

        # Show PCR controls only for PCR datasets
        if dataset_type == 'pcr':
            pcr_controls_style = {'display': 'block'}

    return [view_controls_style, pcr_controls_style]


@callback(
    outputs=[
        Output('lod-info-display', 'children')
    ],
    inputs=[
        Input('lod-type-dropdown', 'value')
    ],
    group="3d_settings"
)
def update_lod_info_display(lod_type: str) -> str:
    """Update LOD information display based on selected LOD type."""
    if not lod_type:
        return ""
    
    lod_descriptions = {
        'continuous': 'Real-time adaptive sampling based on camera distance. Provides smooth performance scaling.',
        'discrete': 'Fixed LOD levels with 2x downsampling per level. Predictable performance.',
        'none': 'No level of detail - shows all points. May impact performance with large datasets.'
    }
    
    return lod_descriptions.get(lod_type, f"Unknown LOD type: {lod_type}")
