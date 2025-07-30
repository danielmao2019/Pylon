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
    
    # All dataset display_datapoint methods have the same signature with optional parameters
    # Call with all parameters since they're all optional in the base class definitions
    return display_func(
        datapoint=datapoint,
        class_labels=class_labels,
        camera_state=camera_state,
        settings_3d=settings_3d
    )
