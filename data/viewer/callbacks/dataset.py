"""Dataset-related callbacks for the viewer."""
from typing import Dict, List, Optional, Union, Any, Literal
from dash import Input, Output, State, html
from dash.exceptions import PreventUpdate
import logging
from data.viewer.states.viewer_state import ViewerEvent
from data.viewer.layout.display.dataset import create_dataset_info_display
from data.viewer.layout.controls.transforms import create_transforms_section
from data.viewer.callbacks.registry import callback, registry

logger = logging.getLogger(__name__)

# Dataset type definitions
DatasetType = Literal['2d_change_detection', '3d_change_detection', 'point_cloud_registration']

# Dataset type mapping
DATASET_TYPE_MAPPING = {
    'point_cloud_registration': 'point_cloud_registration',
    'urb3dcd': '3d_change_detection',
    'slpccd': '3d_change_detection',
    'air_change': '2d_change_detection',
    'cdd': '2d_change_detection',
    'levir_cd': '2d_change_detection',
    'oscd': '2d_change_detection',
    'sysu_cd': '2d_change_detection'
}

def get_dataset_type(dataset_name: str) -> DatasetType:
    """Determine the dataset type from the dataset name."""
    # Extract base name if it contains a path
    base_name = dataset_name.split('/')[-1] if '/' in dataset_name else dataset_name
    
    # Get the dataset type from the mapping
    dataset_type = DATASET_TYPE_MAPPING.get(base_name, '2d_change_detection')
    
    # Special case for PCR datasets that might have a different naming pattern
    if 'pcr' in base_name.lower():
        dataset_type = 'point_cloud_registration'
        
    return dataset_type

@callback(
    outputs=[
        Output('dataset-info', 'data'),
        Output('datapoint-index-slider', 'min'),
        Output('datapoint-index-slider', 'max'),
        Output('datapoint-index-slider', 'value'),
        Output('datapoint-index-slider', 'marks'),
        Output('datapoint-display', 'children', allow_duplicate=True),
        Output('dataset-info-display', 'children'),
        Output('transforms-section', 'children')
    ],
    inputs=[Input('dataset-dropdown', 'value')],
    group="dataset"
)
def load_dataset(dataset_name: Optional[str]) -> List[Union[Dict[str, Any], int, html.Div]]:
    """Load a selected dataset and reset the datapoint slider."""
    logger.info(f"Dataset loading callback triggered with dataset: {dataset_name}")
    
    if dataset_name is None:
        logger.info("No dataset selected")
        registry.viewer.state.reset()
        return [
            {},  # dataset-info
            0,   # min
            0,   # max
            0,   # value
            {},  # marks
            html.Div("No dataset selected."),  # datapoint-display
            create_dataset_info_display(),  # dataset-info-display
            create_transforms_section(),  # transforms-section
        ]

    # Load dataset using dataset manager
    logger.info(f"Attempting to load dataset: {dataset_name}")
    dataset_info = registry.viewer.dataset_manager.load_dataset(dataset_name)
    
    # Determine dataset type
    dataset_type = get_dataset_type(dataset_name)
    dataset_info['type'] = dataset_type
    
    # Set is_3d based on dataset type
    dataset_info['is_3d'] = dataset_type in ['3d_change_detection', 'point_cloud_registration']

    # Update state with dataset info
    logger.info(f"Updating state with dataset info: {dataset_info}")
    registry.viewer.state.update_dataset_info(
        name=dataset_info['name'],
        length=dataset_info['length'],
        class_labels=dataset_info['class_labels'],
        is_3d=dataset_info['is_3d'],
        transforms=dataset_info['transforms']
    )

    # Create slider marks
    marks = {}
    if dataset_info['length'] <= 10:
        marks = {i: str(i) for i in range(dataset_info['length'])}
    else:
        step = max(1, dataset_info['length'] // 10)
        marks = {i: str(i) for i in range(0, dataset_info['length'], step)}
        marks[dataset_info['length'] - 1] = str(dataset_info['length'] - 1)

    # Get initial message
    initial_message = html.Div(f"Dataset '{dataset_name}' loaded successfully with {dataset_info['length']} datapoints. Use the slider to navigate.")
    logger.info("Dataset loaded successfully, returning updated UI components")

    return [
        registry.viewer.state.get_state()['dataset_info'],  # dataset-info
        0,                   # min
        dataset_info['length'] - 1,  # max
        0,                # value
        marks,              # marks
        initial_message,    # datapoint-display
        create_dataset_info_display(registry.viewer.state.get_state()['dataset_info']),  # dataset-info-display
        create_transforms_section(dataset_info['transforms']),  # transforms-section
    ]


@callback(
    outputs=Output('dataset-dropdown', 'options'),
    inputs=[Input('reload-button', 'n_clicks')],
    group="dataset"
)
def reload_datasets(n_clicks: Optional[int]) -> List[Dict[str, str]]:
    """Reload available datasets."""
    if n_clicks is None:
        raise PreventUpdate

    # Get list of available datasets
    available_datasets = registry.viewer.dataset_manager.get_available_datasets()

    # Create options for the dropdown
    return [{'label': name, 'value': name} for name in available_datasets]
