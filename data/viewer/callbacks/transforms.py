"""Transform selection callbacks for the viewer."""
from typing import Dict, List, Optional, Union
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
        Input('3d-settings-store', 'data'),
        Input('camera-state', 'data')
    ],
    states=[
        State('dataset-info', 'data'),
        State('datapoint-index-slider', 'value')
    ],
    group="transforms"
)
def update_datapoint_from_transforms(
    transform_values: List[List[int]],
    settings_3d: Optional[Dict[str, Union[str, int, float, bool]]],
    camera_state: Dict,
    dataset_info: Optional[Dict[str, Union[str, int, bool, Dict]]],
    datapoint_idx: int
) -> List[html.Div]:
    """
    Update the displayed datapoint when transform selections change.
    Also handles 3D point cloud visualization settings.
    """
    logger.info(f"Transform selection callback triggered - Transform values: {transform_values}")

    # Handle case where no dataset is selected (normal UI state)
    if dataset_info is None or dataset_info == {}:
        raise PreventUpdate
    
    # Assert dataset info structure is valid - fail fast if corrupted
    assert dataset_info is not None, "Dataset info must not be None"
    assert dataset_info != {}, "Dataset info must not be empty"
    assert 'name' in dataset_info, f"Dataset info must have 'name' key, got keys: {list(dataset_info.keys())}"
    assert 'type' in dataset_info, f"Dataset info must have 'type' key, got keys: {list(dataset_info.keys())}"

    dataset_name: str = dataset_info['name']
    dataset_type: str = dataset_info['type']
    logger.info(f"Updating datapoint for dataset: {dataset_name}")

    # Get list of selected transform indices
    selected_indices = [
        idx for values in transform_values
        for idx in values  # values will be a list containing the index if checked, empty if not
    ]

    # Get datapoint from backend through registry using kwargs
    datapoint = registry.viewer.backend.get_datapoint(
        dataset_name=dataset_name,
        index=datapoint_idx,
        transform_indices=selected_indices
    )
    
    logger.info(f"Dataset type: {dataset_type}")

    # Extract 3D settings and class labels using centralized configuration
    settings_3d = ViewerSettings.get_3d_settings_with_defaults(settings_3d)
    class_labels = dataset_info.get('class_labels') if dataset_type in ['semseg', '3dcd'] else None

    # Create display using kwargs to prevent parameter ordering issues
    logger.info(f"Creating {dataset_type} display with selected transforms")
    display = create_display(
        dataset_type=dataset_type,
        datapoint=datapoint,
        class_labels=class_labels,
        camera_state=camera_state,
        settings_3d=settings_3d,
    )

    logger.info("Display created successfully with transform selection")
    return [display]
