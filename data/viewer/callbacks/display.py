"""Display-related callbacks for the viewer."""
from typing import Dict, List, Optional, Union, Any
from dash import Input, Output, State, html, ALL
from dash.exceptions import PreventUpdate
from data.viewer.callbacks.registry import callback, registry
from data.viewer.layout.controls.transforms import create_transforms_section
from data.viewer.layout.display.display_2dcd import display_2dcd_datapoint
from data.viewer.layout.display.display_3dcd import display_3dcd_datapoint
from data.viewer.layout.display.display_pcr import display_pcr_datapoint
from data.viewer.layout.display.display_semseg import display_semseg_datapoint
from data.viewer.utils.settings_config import ViewerSettings

import logging
logger = logging.getLogger(__name__)


# Mapping of dataset types to their display functions
DISPLAY_FUNCTIONS = {
    'semseg': display_semseg_datapoint,
    '2dcd': display_2dcd_datapoint,
    '3dcd': display_3dcd_datapoint,
    'pcr': display_pcr_datapoint,
}


def _create_display(
    dataset_type: str,
    display_func: Any,
    datapoint: Dict[str, Any], 
    class_labels: Optional[Dict[int, str]],
    camera_state: Dict[str, Any],
    settings_3d: Dict[str, Union[float, str]]
) -> html.Div:
    """Create display based on dataset type with appropriate parameters."""
    # Define parameter mappings for each dataset type
    display_params = {
        'semseg': {
            'args': (datapoint,),
            'kwargs': {}
        },
        '2dcd': {
            'args': (datapoint,),
            'kwargs': {}
        },
        '3dcd': {
            'args': (datapoint, class_labels, camera_state),
            'kwargs': {
                'point_size': settings_3d['point_size'],
                'point_opacity': settings_3d['point_opacity'],
                'lod_type': settings_3d['lod_type']
            }
        },
        'pcr': {
            'args': (),
            'kwargs': {
                'datapoint': datapoint,
                'camera_state': camera_state,
                'point_size': settings_3d['point_size'],
                'point_opacity': settings_3d['point_opacity'],
                'sym_diff_radius': settings_3d['sym_diff_radius'],
                'corr_radius': settings_3d['corr_radius'],
                'lod_type': settings_3d['lod_type']
            }
        }
    }
    
    if dataset_type not in display_params:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    
    params = display_params[dataset_type]
    return display_func(*params['args'], **params['kwargs'])


@callback(
    outputs=[
        Output('datapoint-display', 'children'),
    ],
    inputs=[
        Input('dataset-info', 'data'),
        Input('datapoint-index-slider', 'value'),
        Input({'type': 'transform-checkbox', 'index': ALL}, 'value'),
        Input('camera-state', 'data'),
        Input('3d-settings-store', 'data')
    ],
    group="display"
)
def update_datapoint(
    dataset_info: Optional[Dict[str, Union[str, int, bool, Dict]]],
    datapoint_idx: int,
    transform_values: List[List[int]],
    camera_state: Dict,
    settings_3d: Optional[Dict[str, Union[str, int, float, bool]]]
) -> List[html.Div]:
    """
    Update the displayed datapoint based on the slider value.
    Also handles 3D point cloud visualization settings.
    """
    logger.info(f"Display update callback triggered - Dataset info: {dataset_info}, Index: {datapoint_idx}")

    if dataset_info is None or dataset_info == {}:
        logger.warning("No dataset info available")
        return [html.Div("No dataset loaded.")]

    dataset_name: str = dataset_info.get('name', 'unknown')
    logger.info(f"Attempting to get dataset: {dataset_name}")

    # Get list of selected transform indices
    selected_indices = [
        idx for values in transform_values
        for idx in values  # values will be a list containing the index if checked, empty if not
    ]

    # Get datapoint from backend through registry
    datapoint = registry.viewer.backend.get_datapoint(dataset_name, datapoint_idx, selected_indices)

    # Get dataset type and display function
    dataset_type = dataset_info['type']  # Will raise KeyError if missing - that's a bug that should be caught
    display_func = DISPLAY_FUNCTIONS[dataset_type]  # Will raise KeyError if unsupported - that's a bug that should be caught
    
    logger.info(f"Dataset type: {dataset_type}")

    # Extract 3D settings and class labels using centralized configuration
    settings = ViewerSettings.get_3d_settings_with_defaults(settings_3d)
    class_labels = dataset_info.get('class_labels') if dataset_type in ['semseg', '3dcd'] else None

    # Call the display function with appropriate parameters
    logger.info(f"Creating {dataset_type} display")
    display = _create_display(dataset_type, display_func, datapoint, class_labels, camera_state, settings)

    logger.info("Display created successfully")
    return [display]


@callback(
    outputs=[
        Output('transforms-section', 'children', allow_duplicate=True),
        Output('datapoint-display', 'children', allow_duplicate=True)
    ],
    inputs=[Input('dataset-info', 'data')],
    states=[
        State('datapoint-index-slider', 'value'),
        State('3d-settings-store', 'data'),
        State('camera-state', 'data')
    ],
    group="display"
)
def update_transforms(dataset_info: Dict[str, Any], datapoint_idx: Optional[int], settings_3d: Optional[Dict[str, Union[str, int, float, bool]]], camera_state: Dict[str, Any]) -> List[Union[html.Div, List[html.Div]]]:
    """Update the transforms section when dataset info changes and display datapoint with all transforms applied."""
    # If no dataset is selected, maintain current state
    if not dataset_info:
        raise PreventUpdate

    dataset_name = dataset_info['name']
    transforms = dataset_info.get('transforms', [])

    # Use index 0 if datapoint_idx is None (first dataset selection)
    if datapoint_idx is None:
        datapoint_idx = 0

    # Create updated transforms section
    transforms_section = create_transforms_section(transforms)

    # Display datapoint with all transforms applied by default
    # Get all transform indices to apply all transforms by default
    all_transform_indices = [transform['index'] for transform in transforms]

    # Get transformed datapoint using backend
    datapoint = registry.viewer.backend.get_datapoint(dataset_name, datapoint_idx, all_transform_indices)

    # Get dataset type and display function  
    dataset_type = dataset_info['type']
    display_func = DISPLAY_FUNCTIONS[dataset_type]  # Will raise KeyError if unsupported - that's a bug that should be caught

    # Extract 3D settings and class labels using centralized configuration
    settings = ViewerSettings.get_3d_settings_with_defaults(settings_3d)
    class_labels = dataset_info.get('class_labels', {}) if dataset_type in ['semseg', '3dcd'] else None

    # Call the display function with current 3D settings
    display = _create_display(dataset_type, display_func, datapoint, class_labels, camera_state, settings)

    return [transforms_section, [display]]
