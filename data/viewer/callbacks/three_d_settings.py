"""3D settings-related callbacks for the viewer."""
from typing import Dict, List, Optional, Union, Any
from dash import Input, Output
import dash
from dash.exceptions import PreventUpdate
from data.viewer.callbacks.registry import callback
from data.viewer.utils.settings_config import ViewerSettings
from data.viewer.utils.debounce import debounce


@callback(
    outputs=[
        Output('3d-settings-store', 'data')
    ],
    inputs=[
        Input('point-size-slider', 'value'),
        Input('point-opacity-slider', 'value'),
        Input('radius-slider', 'value'),
        Input('correspondence-radius-slider', 'value'),
        Input('lod-type-dropdown', 'value'),
        Input('density-slider', 'value')
    ],
    group="3d_settings"
)
@debounce
def update_3d_settings(
    point_size: float,
    point_opacity: float,
    sym_diff_radius: float,
    corr_radius: float,
    lod_type: str,
    density_percentage: int
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
        density_percentage: Percentage of points to display when LOD is 'none' (1-100)
    
    Returns:
        List containing a dictionary of all 3D settings for UI store
    """
    if point_size is None or point_opacity is None:
        # Thread-safe return instead of raising PreventUpdate in debounced context
        return [dash.no_update]

    # Create settings dictionary with provided values
    raw_settings = {
        'point_size': point_size,
        'point_opacity': point_opacity,
        'sym_diff_radius': sym_diff_radius,
        'corr_radius': corr_radius,
        'lod_type': lod_type,
        'density_percentage': density_percentage
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

    assert 'requires_3d_visualization' in dataset_info, f"dataset_info missing required 'requires_3d_visualization' key, got keys: {list(dataset_info.keys())}"
    assert 'type' in dataset_info, f"dataset_info missing required 'type' key, got keys: {list(dataset_info.keys())}"
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


@callback(
    outputs=[Output('lod-info-display', 'children')],
    inputs=[
        Input('lod-type-dropdown', 'value')
    ],
    group="3d_settings"
)
def update_lod_info_display(lod_type: str) -> List[str]:
    """Update LOD information display based on selected LOD type."""
    if not lod_type:
        return [""]
    
    lod_descriptions = {
        'continuous': 'Real-time adaptive sampling based on camera distance. Provides smooth performance scaling.',
        'discrete': 'Fixed LOD levels with 2x downsampling per level. Predictable performance.',
        'none': 'No level of detail - shows all points. Use density control to adjust point count.'
    }
    
    return [lod_descriptions.get(lod_type, f"Unknown LOD type: {lod_type}")]


@callback(
    outputs=[Output('density-controls', 'style')],
    inputs=[Input('lod-type-dropdown', 'value')],
    group="3d_settings"
)
def update_density_controls_visibility(lod_type: str) -> List[Dict[str, str]]:
    """Show density controls only when LOD type is 'none'."""
    if lod_type == 'none':
        return [{'display': 'block', 'margin-top': '20px'}]
    else:
        return [{'display': 'none'}]
