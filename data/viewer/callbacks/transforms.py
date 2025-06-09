"""Transform-related callbacks for the viewer."""
from typing import Dict, List, Optional, Union, Any
from dash import Input, Output, State, ALL, html
from dash.exceptions import PreventUpdate
from data.viewer.layout.display.display_semseg import display_semseg_datapoint
from data.viewer.layout.display.display_2dcd import display_2dcd_datapoint
from data.viewer.layout.display.display_3dcd import display_3dcd_datapoint
from data.viewer.layout.display.display_pcr import display_pcr_datapoint
from data.viewer.layout.controls.transforms import create_transforms_section
from data.viewer.callbacks.registry import callback, registry


@callback(
    outputs=Output('datapoint-display', 'children', allow_duplicate=True),
    inputs=[Input({'type': 'transform-checkbox', 'index': ALL}, 'value')],
    states=[
        State('dataset-info', 'data'),
        State('datapoint-index-slider', 'value')
    ],
    group="transforms"
)
def apply_transforms(
    transform_values: List[List[int]],
    dataset_info: Optional[Dict[str, Any]],
    datapoint_idx: int
) -> List[html.Div]:
    """Apply the selected transforms to the current datapoint."""
    if not dataset_info or 'name' not in dataset_info:
        raise PreventUpdate

    dataset_name = dataset_info['name']

    # Get list of selected transform indices
    selected_indices = [
        idx for values in transform_values
        for idx in values  # values will be a list containing the index if checked, empty if not
    ]

    # Get transformed datapoint using dataset manager
    datapoint = registry.viewer.dataset_manager.get_datapoint(dataset_name, datapoint_idx, selected_indices)
    assert isinstance(datapoint, dict), f"Datapoint must be a dictionary. Got {type(datapoint)}."
    assert 'inputs' in datapoint, f"Datapoint missing 'inputs' field. Got {datapoint.keys()}."
    assert isinstance(datapoint['inputs'], dict), f"Datapoint 'inputs' must be a dictionary. Got {type(datapoint['inputs'])}."

    # Get dataset type and determine display function
    dataset_type = dataset_info['type']

    # Display the transformed datapoint
    if dataset_type == 'semseg':
        display = display_semseg_datapoint(datapoint)
    elif dataset_type == '2dcd':
        display = display_2dcd_datapoint(datapoint, class_names=dataset_info['class_labels'])
    elif dataset_type == '3dcd':
        display = display_3dcd_datapoint(datapoint, class_names=dataset_info['class_labels'])
    elif dataset_type == 'pcr':
        display = display_pcr_datapoint(datapoint)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

    return [display]


@callback(
    outputs=[
        Output('transforms-section', 'children', allow_duplicate=True),
        Output('transforms-store', 'data')
    ],
    inputs=[Input('dataset-info', 'data')],
    group="transforms"
)
def update_transforms(dataset_info: Dict[str, Any]) -> List[Union[html.Div, Dict[str, Any]]]:
    """Update the transforms section when dataset info changes."""
    # If no dataset is selected, maintain current state
    if not dataset_info:
        raise PreventUpdate

    assert isinstance(dataset_info, dict), f"Dataset info must be a dictionary. Got {type(dataset_info)}."
    assert 'transforms' in dataset_info, f"Dataset info missing 'transforms' field. Got {dataset_info.keys()}."
    transforms = dataset_info['transforms']
    assert isinstance(transforms, list), f"Transforms must be a list. Got {type(transforms)}."

    # Update state with new transforms
    registry.viewer.state.update_transforms(transforms)

    # Create updated transforms section
    transforms_section = create_transforms_section(transforms)

    return [
        transforms_section,
        registry.viewer.state.get_state()['transforms']
    ]
