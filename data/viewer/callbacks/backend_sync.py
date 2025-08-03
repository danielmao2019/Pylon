"""Backend synchronization callbacks for the viewer.

This module contains callbacks that are responsible for syncing UI state changes
with the backend state. These callbacks follow the pure backend update pattern:
- They listen to UI state changes (dcc.Store components)
- They update backend state as their primary responsibility
- They don't return UI components (use prevent_update or minimal outputs)

This separates backend state management from UI rendering for cleaner architecture.
"""
from typing import Dict, List, Optional, Union, Any
from dash import Input, Output, State
import dash
from dash.exceptions import PreventUpdate
from data.viewer.callbacks.registry import callback, registry
from data.viewer.utils.settings_config import ViewerSettings
from data.viewer.utils.debounce import debounce

import logging
logger = logging.getLogger(__name__)


@callback(
    outputs=[
        Output('backend-sync-3d-settings', 'data')  # Dummy output for sync signal
    ],
    inputs=[
        Input('3d-settings-store', 'data')
    ],
    group="backend_sync"
)
@debounce
def sync_3d_settings_to_backend(settings_3d: Optional[Dict[str, Union[str, int, float, bool]]]) -> List[Dict[str, Any]]:
    """Sync 3D settings from UI store to backend state.
    
    This callback is purely for backend synchronization - it doesn't render UI.
    It listens to changes in the 3D settings store and updates backend accordingly.
    """
    if not settings_3d:
        # Thread-safe return instead of raising PreventUpdate in debounced context
        return [dash.no_update]
    
    logger.info(f"Syncing 3D settings to backend: {settings_3d}")
    
    # Validate and apply defaults using centralized configuration
    validated_settings = ViewerSettings.validate_3d_settings(
        ViewerSettings.get_3d_settings_with_defaults(settings_3d)
    )
    
    # Update backend state with validated settings
    registry.viewer.backend.update_state(**validated_settings)
    
    logger.info("3D settings synced to backend successfully")
    
    # Return minimal sync signal (not used for UI)
    return [{'synced': True, 'timestamp': str(id(validated_settings))}]


@callback(
    outputs=[
        Output('backend-sync-dataset', 'data')  # Dummy output for sync signal
    ],
    inputs=[
        Input('dataset-info', 'data')
    ],
    group="backend_sync"  
)
def sync_dataset_to_backend(dataset_info: Optional[Dict[str, Union[str, int, bool, Dict]]]) -> List[Dict[str, Any]]:
    """Sync dataset info from UI store to backend state.
    
    This callback is purely for backend synchronization - it doesn't render UI.
    It listens to changes in dataset info and updates backend accordingly.
    """
    # Handle case where no dataset is selected (normal UI state)
    if dataset_info is None or dataset_info == {}:
        # Handle dataset deselection or empty dataset info
        logger.info("No dataset selected - clearing backend state")
        registry.viewer.backend.current_dataset = None
        return [{'synced': True, 'dataset': None}]
    
    # Assert dataset info structure is valid - fail fast if corrupted
    assert dataset_info is not None, "Dataset info must not be None"
    assert dataset_info != {}, "Dataset info must not be empty"
    assert 'name' in dataset_info, f"Dataset info must have 'name' key, got keys: {list(dataset_info.keys())}"
    
    dataset_name: str = dataset_info['name']
    logger.info(f"Syncing dataset to backend: {dataset_name}")
    
    # Update backend state with dataset info
    registry.viewer.backend.update_state(
        current_dataset=dataset_name,
        current_index=0  # Reset to first datapoint when dataset changes
    )
    
    logger.info("Dataset info synced to backend successfully")
    
    # Return minimal sync signal (not used for UI)
    return [{'synced': True, 'dataset': dataset_name}]


@callback(
    outputs=[
        Output('backend-sync-navigation', 'data')  # Dummy output for sync signal
    ],
    inputs=[
        Input('datapoint-index-slider', 'value')
    ],
    states=[
        State('dataset-info', 'data')
    ],
    group="backend_sync"
)
@debounce
def sync_navigation_to_backend(
    datapoint_idx: int,
    dataset_info: Optional[Dict[str, Union[str, int, bool, Dict]]]
) -> List[Dict[str, Any]]:
    """Sync navigation index from UI to backend state.
    
    This callback is purely for backend synchronization - it doesn't render UI.
    It listens to changes in the navigation index and updates backend accordingly.
    """
    # Handle case where no dataset is selected (normal UI state)
    if datapoint_idx is None or dataset_info is None or dataset_info == {}:
        # Thread-safe return instead of raising PreventUpdate in debounced context
        return [dash.no_update]
    
    # Assert dataset info structure is valid - fail fast if corrupted
    assert dataset_info is not None, "Dataset info must not be None"
    assert dataset_info != {}, "Dataset info must not be empty"
    assert 'name' in dataset_info, f"Dataset info must have 'name' key, got keys: {list(dataset_info.keys())}"
    
    logger.info(f"Syncing navigation index to backend: {datapoint_idx}")
    
    # Update backend state with current index
    registry.viewer.backend.update_state(current_index=datapoint_idx)
    
    logger.info("Navigation index synced to backend successfully")
    
    # Return minimal sync signal (not used for UI)
    return [{'synced': True, 'index': datapoint_idx}]
