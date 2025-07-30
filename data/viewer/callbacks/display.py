"""Display utility module for the viewer.

This module provides centralized display creation utilities used by other callback modules.
The actual display update callbacks are distributed across:
- dataset.py: Dataset selection triggered display updates
- transforms.py: Transform checkbox triggered display updates  
- navigation.py: Datapoint navigation triggered display updates

This module contains only shared utilities, no actual callbacks.
"""
from typing import Dict, Optional, Union, Any
from dash import html
from data.viewer.layout.display.display_2dcd import display_2dcd_datapoint
from data.viewer.layout.display.display_3dcd import display_3dcd_datapoint
from data.viewer.layout.display.display_pcr import display_pcr_datapoint
from data.viewer.layout.display.display_semseg import display_semseg_datapoint


# Mapping of dataset types to their display functions
DISPLAY_FUNCTIONS = {
    'semseg': display_semseg_datapoint,
    '2dcd': display_2dcd_datapoint,
    '3dcd': display_3dcd_datapoint,
    'pcr': display_pcr_datapoint,
}


def create_display(
    display_func: callable,
    datapoint: Dict[str, Any], 
    class_labels: Optional[Dict[int, str]],
    camera_state: Dict[str, Any],
    settings_3d: Dict[str, Union[float, str]]
) -> html.Div:
    """Create display using the provided display function.
    
    This is a shared utility function used by multiple callback modules.
    The caller is responsible for providing the appropriate display function.
    
    Args:
        display_func: Display function to use for creating the visualization
        datapoint: Dictionary containing inputs, labels, and meta_info
        class_labels: Optional dictionary mapping class indices to label names
        camera_state: Dictionary containing camera position state
        settings_3d: Dictionary containing 3D visualization settings
        
    Returns:
        html.Div containing the visualization
    """
    # Input validation with detailed error messages
    assert callable(display_func), f"display_func must be callable, got {type(display_func)}"
    assert isinstance(datapoint, dict), f"datapoint must be dict, got {type(datapoint)}"
    assert datapoint != {}, "datapoint must not be empty"
    assert isinstance(camera_state, dict), f"camera_state must be dict, got {type(camera_state)}"
    assert isinstance(settings_3d, dict), f"settings_3d must be dict, got {type(settings_3d)}"
    assert class_labels is None or isinstance(class_labels, dict), f"class_labels must be dict or None, got {type(class_labels)}"
    
    # Get function name for parameter determination
    func_name = display_func.__name__ if hasattr(display_func, '__name__') else str(display_func)
    
    # Call the display function with appropriate parameters based on its signature
    if func_name == 'display_semseg_datapoint':
        # Semantic segmentation: only needs datapoint
        return display_func(datapoint=datapoint)
        
    elif func_name == 'display_2dcd_datapoint':
        # 2D change detection: only needs datapoint
        return display_func(datapoint=datapoint)
        
    elif func_name == 'display_3dcd_datapoint':
        # 3D change detection: needs datapoint, class_labels, camera_state, and 3D settings
        return display_func(
            datapoint=datapoint,
            class_names=class_labels,
            camera_state=camera_state,
            point_size=settings_3d['point_size'],
            point_opacity=settings_3d['point_opacity'],
            lod_type=settings_3d['lod_type'],
            density_percentage=settings_3d['density_percentage']
        )
        
    elif func_name == 'display_pcr_datapoint':
        # Point cloud registration: needs specific parameters
        return display_func(
            datapoint=datapoint,
            camera_state=camera_state,
            point_size=settings_3d['point_size'],
            point_opacity=settings_3d['point_opacity'],
            sym_diff_radius=settings_3d['sym_diff_radius'],
            corr_radius=settings_3d['corr_radius'],
            lod_type=settings_3d['lod_type'],
            density_percentage=settings_3d['density_percentage']
        )
    else:
        # For custom display methods from dataset classes, call with datapoint only
        # Dataset classes that have custom display methods should handle all their own parameters
        return display_func(datapoint=datapoint)
