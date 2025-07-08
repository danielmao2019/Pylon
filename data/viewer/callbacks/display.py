"""Pure display update callbacks for the viewer.

This module handles only display rendering updates triggered by:
- Dataset info changes
- Datapoint index changes  
- Camera state changes
- 3D settings changes

Other callback types are handled in separate modules:
- dataset.py: Dataset selection and transforms section updates
- transforms.py: Transform checkbox selection
- navigation.py: Datapoint navigation (prev/next buttons)
- three_d_settings.py: 3D control sliders and visibility
- camera.py: Camera pose changes from 3D graph interactions
"""
from typing import Dict, List, Optional, Union, Any
from dash import Input, Output, State, html
from dash.exceptions import PreventUpdate
from data.viewer.callbacks.registry import callback, registry
from data.viewer.layout.display.display_2dcd import display_2dcd_datapoint
from data.viewer.layout.display.display_3dcd import display_3dcd_datapoint
from data.viewer.layout.display.display_pcr import display_pcr_datapoint
from data.viewer.layout.display.display_semseg import display_semseg_datapoint
from data.viewer.utils.settings_config import ViewerSettings

import logging
logger = logging.getLogger(__name__)


# Mapping of dataset types to their display functions
DISPLAY_FUNCTIONS = {
    'semseg': display_semseg_datapoint,
    '2dcd': display_2dcd_datapoint,
    '3dcd': display_3dcd_datapoint,
    'pcr': display_pcr_datapoint,
}


def _create_display(
    dataset_type: str,
    display_func: Any,
    datapoint: Dict[str, Any], 
    class_labels: Optional[Dict[int, str]],
    camera_state: Dict[str, Any],
    settings_3d: Dict[str, Union[float, str]]
) -> html.Div:
    """Create display based on dataset type with appropriate parameters."""
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
    
    if dataset_type not in display_params:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    
    params = display_params[dataset_type]
    return display_func(*params['args'], **params['kwargs'])




