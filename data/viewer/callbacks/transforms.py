"""Transform selection callbacks for the viewer."""
from typing import Dict, List, Optional, Union, Any
from dash import Input, Output, State, html, ALL
from dash.exceptions import PreventUpdate
from data.viewer.callbacks.registry import callback, registry
from data.viewer.callbacks.display import create_display
from data.viewer.utils.settings_config import ViewerSettings

import logging
logger = logging.getLogger(__name__)




@callback(
    outputs=[
        Output('datapoint-display', 'children'),
    ],
    inputs=[
        Input({'type': 'transform-checkbox', 'index': ALL}, 'value'),
    ],
    states=[
        State('dataset-info', 'data'),
        State('datapoint-index-slider', 'value'),
        State('camera-state', 'data'),
        State('3d-settings-store', 'data')
    ],
    group="transforms"
)
def update_datapoint_from_transforms(
    transform_values: List[List[int]],
    dataset_info: Optional[Dict[str, Union[str, int, bool, Dict]]],
    datapoint_idx: int,
    camera_state: Dict,
    settings_3d: Optional[Dict[str, Union[str, int, float, bool]]]
) -> List[html.Div]:
    """
    Update the displayed datapoint when transform selections change.
    Also handles 3D point cloud visualization settings.
    """
    logger.info(f"Transform selection callback triggered - Transform values: {transform_values}")

    if dataset_info is None or dataset_info == {}:
        logger.warning("No dataset info available")
        return [html.Div("No dataset loaded.")]

    dataset_name: str = dataset_info.get('name', 'unknown')
    logger.info(f"Updating datapoint for dataset: {dataset_name}")

    # Get list of selected transform indices
    selected_indices = [
        idx for values in transform_values
        for idx in values  # values will be a list containing the index if checked, empty if not
    ]

    # Get datapoint from backend through registry
    datapoint = registry.viewer.backend.get_datapoint(dataset_name, datapoint_idx, selected_indices)

    # Get dataset type and create display
    dataset_type = dataset_info['type']  # Will raise KeyError if missing - that's a bug that should be caught
    
    logger.info(f"Dataset type: {dataset_type}")

    # Extract 3D settings and class labels using centralized configuration
    settings = ViewerSettings.get_3d_settings_with_defaults(settings_3d)
    class_labels = dataset_info.get('class_labels') if dataset_type in ['semseg', '3dcd'] else None

    # Call the display function with appropriate parameters
    logger.info(f"Creating {dataset_type} display with selected transforms")
    display = create_display(dataset_type, datapoint, class_labels, camera_state, settings)

    logger.info("Display created successfully with transform selection")
    return [display]
