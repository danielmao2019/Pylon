"""Segmentation color shuffle callbacks for the viewer."""
from typing import Dict, List, Optional, Any
from dash import Input, Output, State, ALL, ctx
from dash.exceptions import PreventUpdate
from data.viewer.callbacks.registry import callback, registry
import dash

import logging
logger = logging.getLogger(__name__)


@callback(
    outputs=[
        Output('datapoint-display', 'children', allow_duplicate=True),
        Output('segmentation-color-seed', 'data', allow_duplicate=True),
    ],
    inputs=[
        Input({'type': 'seg-color-shuffle', 'index': ALL}, 'n_clicks')
    ],
    states=[
        State('segmentation-color-seed', 'data'),
        State('datapoint-index-slider', 'value'),
        State('3d-settings-store', 'data'),
        State('camera-state', 'data'),
        State('dataset-info', 'data'),
        State('transforms-store', 'data')
    ],
    group="segmentation_color"
)
def shuffle_segmentation_colors(
    n_clicks_list: List[Optional[int]],
    current_seed: int,
    datapoint_idx: int,
    settings_3d: Optional[Dict[str, Any]],
    camera_state: Dict,
    dataset_info: Optional[Dict[str, Any]],
    transforms_store: Optional[Dict[str, Any]]
) -> List[Any]:
    """Shuffle the colors for segmentation display.
    
    Args:
        n_clicks_list: List of n_clicks values for each shuffle button
        current_seed: Current color seed value
        datapoint_idx: Current datapoint index
        settings_3d: 3D settings (if applicable)
        camera_state: Camera state (if applicable) 
        dataset_info: Dataset information
        
    Returns:
        Tuple of (updated datapoint display, new color seed)
        
    Raises:
        PreventUpdate: If no button has been clicked
    """
    # CRITICAL: Input validation with fail-fast assertions
    assert isinstance(n_clicks_list, list), f"n_clicks_list must be list, got {type(n_clicks_list)}"
    
    # Check if any button was clicked
    if not any(clicks is not None and clicks > 0 for clicks in n_clicks_list):
        raise PreventUpdate
    
    # Handle case where no dataset is selected
    if dataset_info is None or dataset_info == {} or transforms_store is None or transforms_store == {}:
        raise PreventUpdate
    
    # Assert dataset info structure is valid
    assert dataset_info is not None, "Dataset info must not be None"
    assert dataset_info != {}, "Dataset info must not be empty"
    assert 'name' in dataset_info, f"Dataset info must have 'name' key, got keys: {list(dataset_info.keys())}"
    
    # Extract the triggered component information
    assert ctx.triggered, "Callback context must have triggered components"
    assert len(ctx.triggered) > 0, "Must have at least one triggered component"
    
    import json
    triggered_prop_id = ctx.triggered[0]['prop_id']
    assert isinstance(triggered_prop_id, str), f"Expected string prop_id, got {type(triggered_prop_id)}"
    
    # Parse the component ID
    component_id_str = triggered_prop_id.split('.')[0]
    component_id = json.loads(component_id_str)
    assert isinstance(component_id, dict), f"Expected dict component ID, got {type(component_id)}"
    assert 'index' in component_id, f"Component ID must have 'index', got keys: {list(component_id.keys())}"
    
    triggered_index = component_id['index']
    assert isinstance(triggered_index, int), f"Expected int index, got {type(triggered_index)}"
    
    # Increment the color seed
    new_seed = current_seed + 1
    logger.info(f"Shuffled colors for component {triggered_index}, new seed: {new_seed}")
    
    # Now we need to refresh the display
    # Get dataset info
    dataset_name: str = dataset_info['name']
    
    # CRITICAL: Transform store must contain current dataset's selected transforms
    assert transforms_store is not None, "transforms_store must not be None"
    assert isinstance(transforms_store, dict), f"transforms_store must be dict, got {type(transforms_store)}"
    assert 'dataset_name' in transforms_store, f"transforms_store missing 'dataset_name', got keys: {list(transforms_store.keys())}"
    assert 'selected_indices' in transforms_store, f"transforms_store missing 'selected_indices', got keys: {list(transforms_store.keys())}"
    assert transforms_store['dataset_name'] == dataset_name, f"Dataset mismatch: store has '{transforms_store['dataset_name']}', expected '{dataset_name}'"
    
    selected_indices = transforms_store['selected_indices']
    assert isinstance(selected_indices, list), f"selected_indices must be list, got {type(selected_indices)}"
    assert all(isinstance(idx, int) for idx in selected_indices), f"All selected indices must be int, got {selected_indices}"
    
    logger.info(f"Using selected transforms: {selected_indices}")
    
    # Get datapoint from backend
    datapoint = registry.viewer.backend.get_datapoint(
        dataset_name=dataset_name,
        index=datapoint_idx,
        transform_indices=selected_indices
    )
    
    # Get display settings
    from data.viewer.utils.settings_config import ViewerSettings
    settings_3d = ViewerSettings.get_3d_settings_with_defaults(settings_3d)
    
    # Get dataset instance
    dataset_instance = registry.viewer.backend.get_dataset_instance(dataset_name=dataset_name)
    class_labels = dataset_instance.class_labels if hasattr(dataset_instance, 'class_labels') and dataset_instance.class_labels else None
    
    # Check camera state
    default_camera_state = {
        'eye': {'x': 1.5, 'y': 1.5, 'z': 1.5},
        'center': {'x': 0, 'y': 0, 'z': 0}, 
        'up': {'x': 0, 'y': 0, 'z': 1}
    }
    
    final_camera_state = camera_state
    if camera_state == default_camera_state:
        if ('meta_info' in datapoint and 
            'camera_pose' in datapoint['meta_info'] and 
            'camera_intrinsics' in datapoint['meta_info']):
            final_camera_state = None
    
    # Create display with new colors
    display_func = dataset_instance.display_datapoint
    display = display_func(
        datapoint=datapoint,
        class_labels=class_labels,
        camera_state=final_camera_state,
        settings_3d=settings_3d,
        color_seed=new_seed
    )
    
    logger.info("Segmentation colors shuffled and display updated")
    return [display, new_seed]