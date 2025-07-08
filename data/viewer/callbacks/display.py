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
    dataset_type: str,
    datapoint: Dict[str, Any], 
    class_labels: Optional[Dict[int, str]],
    camera_state: Dict[str, Any],
    settings_3d: Dict[str, Union[float, str]]
) -> html.Div:
    """Create display based on dataset type with appropriate parameters.
    
    This is a shared utility function used by multiple callback modules.
    """
    display_func = DISPLAY_FUNCTIONS.get(dataset_type)
    if not display_func:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    
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
    
    params = display_params[dataset_type]
    return display_func(*params['args'], **params['kwargs'])
