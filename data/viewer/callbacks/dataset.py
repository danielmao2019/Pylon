"""Dataset-related callbacks for the viewer."""
import logging
from typing import Dict, List, Optional, Union, Any

from dash import Input, Output, html
from dash.exceptions import PreventUpdate

from data.viewer.callbacks.registry import callback, registry
from data.viewer.layout.controls.transforms import create_transforms_section
from data.viewer.layout.display.dataset import create_dataset_info_display

logger = logging.getLogger(__name__)

# Constants
TYPE_LABELS = {
    'semseg': 'Semantic Segmentation',
    '2dcd': '2D Change Detection', 
    '3dcd': '3D Change Detection',
    'pcr': 'Point Cloud Registration'
}


# =========================================================================
# Dataset Group Management (Hierarchical Dropdown - Top Level)
# =========================================================================

@callback(
    outputs=Output('dataset-group-dropdown', 'options'),
    inputs=[Input('reload-button', 'n_clicks')],
    group="dataset"
)
def reload_dataset_groups(n_clicks: Optional[int]) -> List[Dict[str, str]]:
    """Reload available dataset groups for hierarchical dropdown."""
    if n_clicks is None:
        raise PreventUpdate

    hierarchical_datasets = registry.viewer.backend.get_available_datasets_hierarchical()
    
    # Create group dropdown options
    options = []
    for dataset_type in sorted(hierarchical_datasets.keys()):
        label = TYPE_LABELS.get(dataset_type, dataset_type.upper())
        dataset_count = len(hierarchical_datasets[dataset_type])
        options.append({
            'label': f"{label} ({dataset_count} datasets)",
            'value': dataset_type
        })
    
    return options


# =========================================================================
# Dataset Selection Management (Hierarchical Dropdown - Second Level)
# =========================================================================

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

    hierarchical_datasets = registry.viewer.backend.get_available_datasets_hierarchical()
    
    if selected_group not in hierarchical_datasets:
        return [[], True, "No datasets found for this category", None]

    # Create options for the selected group
    options = []
    for dataset_key, dataset_name in sorted(hierarchical_datasets[selected_group].items()):
        options.append({'label': dataset_name, 'value': dataset_key})

    placeholder = f"Choose from {len(options)} available datasets..."
    return [options, False, placeholder, None]


# =========================================================================
# Dataset Loading and UI Management
# =========================================================================

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
    """Load a selected dataset and update all related UI components.
    
    This is a PURE UI callback - it only updates UI components.
    Backend synchronization happens in backend_sync.py automatically.
    """
    logger.info(f"Dataset loading callback triggered with dataset: {dataset_key}")

    if dataset_key is None:
        logger.info("No dataset selected")
        return _create_empty_dataset_state()

    # Load dataset using backend (read-only operation)
    logger.info(f"Attempting to load dataset: {dataset_key}")
    dataset_info = registry.viewer.backend.load_dataset(dataset_key)

    # Create slider marks for navigation
    marks = _create_slider_marks(dataset_info['length'])

    # Create success message
    success_message = html.Div(
        f"Dataset '{dataset_key}' loaded successfully with {dataset_info['length']} datapoints. "
        "Use the slider to navigate."
    )
    
    logger.info("Dataset loaded successfully, returning updated UI components")

    return [
        dataset_info,                                           # dataset-info
        0,                                                      # slider min
        dataset_info['length'] - 1,                            # slider max
        0,                                                      # slider value
        marks,                                                  # slider marks
        success_message,                                        # datapoint-display
        create_dataset_info_display(dataset_info),             # dataset-info-display
        create_transforms_section(dataset_info['transforms']), # transforms-section
    ]


@callback(
    outputs=[Output('transforms-section', 'children', allow_duplicate=True)],
    inputs=[Input('dataset-info', 'data')],
    group="dataset"
)
def update_transforms_section(dataset_info: Dict[str, Any]) -> List[html.Div]:
    """Update the transforms section when dataset info changes."""
    if not dataset_info:
        raise PreventUpdate

    transforms = dataset_info.get('transforms', [])
    transforms_section = create_transforms_section(transforms)
    
    return [transforms_section]


# =========================================================================
# Helper Functions
# =========================================================================

def _create_empty_dataset_state() -> List[Union[Dict[str, Any], int, html.Div]]:
    """Create UI state for when no dataset is selected."""
    return [
        {},                              # dataset-info (empty)
        0,                               # slider min
        0,                               # slider max
        0,                               # slider value
        {},                              # slider marks (empty)
        html.Div("No dataset selected."), # datapoint-display
        create_dataset_info_display(),    # dataset-info-display
        create_transforms_section(),      # transforms-section
    ]


def _create_slider_marks(dataset_length: int) -> Dict[int, str]:
    """Create slider marks based on dataset length."""
    if dataset_length <= 10:
        # Show all indices for small datasets
        return {i: str(i) for i in range(dataset_length)}
    else:
        # Show ~10 evenly spaced marks for larger datasets
        step = max(1, dataset_length // 10)
        marks = {i: str(i) for i in range(0, dataset_length, step)}
        # Always include the last index
        marks[dataset_length - 1] = str(dataset_length - 1)
        return marks
