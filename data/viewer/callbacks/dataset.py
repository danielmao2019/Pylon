"""Dataset-related callbacks for the viewer."""
from typing import Dict, List, Optional, Union, Any
from dash import Input, Output, html
from dash.exceptions import PreventUpdate
from data.viewer.callbacks.registry import callback, registry
from data.viewer.layout.controls.transforms import create_transforms_section
from data.viewer.layout.display.dataset import create_dataset_info_display

import logging
logger = logging.getLogger(__name__)


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
def load_dataset(dataset_key: Optional[str]) -> List[Union[Dict[str, Any], int, html.Div]]:
    """Load a selected dataset and reset the datapoint slider.
    
    This is now a PURE UI callback - it only updates UI components.
    Backend synchronization happens in backend_sync.py automatically.
    """
    logger.info(f"Dataset loading callback triggered with dataset: {dataset_key}")

    if dataset_key is None:
        logger.info("No dataset selected")
        # PURE UI PATTERN: Only return UI components
        # Backend sync happens automatically in backend_sync.py
        return [
            {},  # dataset-info (empty triggers backend sync to clear state)
            0,   # min
            0,   # max
            0,   # value
            {},  # marks
            html.Div("No dataset selected."),  # datapoint-display
            create_dataset_info_display(),  # dataset-info-display
            create_transforms_section(),  # transforms-section
        ]

    # Load dataset using backend (read-only operation)
    logger.info(f"Attempting to load dataset: {dataset_key}")
    dataset_info = registry.viewer.backend.load_dataset(dataset_key)

    # PURE UI PATTERN: Only prepare UI components
    # Backend sync happens automatically in backend_sync.py when dataset-info changes

    # Create slider marks
    marks = {}
    if dataset_info['length'] <= 10:
        marks = {i: str(i) for i in range(dataset_info['length'])}
    else:
        step = max(1, dataset_info['length'] // 10)
        marks = {i: str(i) for i in range(0, dataset_info['length'], step)}
        marks[dataset_info['length'] - 1] = str(dataset_info['length'] - 1)

    # Get initial message
    initial_message = html.Div(f"Dataset '{dataset_key}' loaded successfully with {dataset_info['length']} datapoints. Use the slider to navigate.")
    logger.info("Dataset loaded successfully, returning updated UI components")

    return [
        dataset_info,  # dataset-info
        0,                   # min
        dataset_info['length'] - 1,  # max
        0,                # value
        marks,              # marks
        initial_message,    # datapoint-display
        create_dataset_info_display(dataset_info),  # dataset-info-display
        create_transforms_section(dataset_info['transforms']),  # transforms-section
    ]


@callback(
    outputs=[
        Output('transforms-section', 'children', allow_duplicate=True),
    ],
    inputs=[Input('dataset-info', 'data')],
    group="dataset"
)
def update_transforms_section(dataset_info: Dict[str, Any]) -> List[html.Div]:
    """Update the transforms section when dataset info changes."""
    # If no dataset is selected, maintain current state
    if not dataset_info:
        raise PreventUpdate

    transforms = dataset_info.get('transforms', [])

    # Create updated transforms section
    transforms_section = create_transforms_section(transforms)

    return [transforms_section]


@callback(
    outputs=[
        Output('dataset-dropdown', 'options'),
        Output('dataset-dropdown', 'disabled'),
        Output('dataset-dropdown', 'placeholder'),
        Output('dataset-dropdown', 'value')
    ],
    inputs=[Input('dataset-group-dropdown', 'value')],
    group="dataset"
)
def update_dataset_options(selected_group: Optional[str]) -> List[Union[List[Dict[str, str]], bool, str, None]]:
    """Update dataset dropdown options based on selected group."""
    if selected_group is None:
        return [[], True, "First select a category above...", None]

    # Get hierarchical datasets
    hierarchical_datasets = registry.viewer.backend.get_available_datasets_hierarchical()
    
    if selected_group not in hierarchical_datasets:
        return [[], True, "No datasets found for this category", None]

    # Create options for the selected group
    options = []
    for dataset_key, dataset_name in sorted(hierarchical_datasets[selected_group].items()):
        options.append({'label': dataset_name, 'value': dataset_key})

    return [options, False, f"Choose from {len(options)} available datasets...", None]


@callback(
    outputs=Output('dataset-group-dropdown', 'options'),
    inputs=[Input('reload-button', 'n_clicks')],
    group="dataset"  
)
def reload_dataset_groups(n_clicks: Optional[int]) -> List[Dict[str, str]]:
    """Reload available dataset groups for hierarchical dropdown."""
    if n_clicks is None:
        raise PreventUpdate

    # Get hierarchical datasets
    hierarchical_datasets = registry.viewer.backend.get_available_datasets_hierarchical()
    
    # Create type labels mapping
    type_labels = {
        'semseg': 'Semantic Segmentation',
        '2dcd': '2D Change Detection', 
        '3dcd': '3D Change Detection',
        'pcr': 'Point Cloud Registration'
    }
    
    # Create group dropdown options
    options = []
    for dataset_type in sorted(hierarchical_datasets.keys()):
        label = type_labels.get(dataset_type, dataset_type.upper())
        options.append({'label': f"{label} ({len(hierarchical_datasets[dataset_type])} datasets)", 'value': dataset_type})
    
    return options
