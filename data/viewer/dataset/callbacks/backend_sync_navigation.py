"""Backend synchronization callbacks for the viewer.

This module contains callbacks that are responsible for syncing UI state changes
with the backend state. These callbacks follow the pure backend update pattern:
- They listen to UI state changes (dcc.Store components)
- They update backend state as their primary responsibility
- They don't return UI components (use prevent_update or minimal outputs)

This separates backend state management from UI rendering for cleaner architecture.
"""

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import dash
from dash.dependencies import Input, Output, State

from data.viewer.utils.debounce import debounce

if TYPE_CHECKING:
    from data.viewer.dataset.viewer import DatasetViewer

logger = logging.getLogger(__name__)


def register_backend_sync_navigation_callbacks(
    app: dash.Dash, viewer: "DatasetViewer"
) -> None:
    # Input validations
    assert isinstance(app, dash.Dash), f"app must be Dash, got {type(app)}"
    assert hasattr(viewer, "backend"), f"viewer must expose backend, got {type(viewer)}"

    @app.callback(
        [Output('backend-sync-navigation', 'data')],
        [Input('datapoint-index-slider', 'value')],
        [State('dataset-info', 'data')],
    )
    @debounce
    def sync_navigation_to_backend(
        datapoint_idx: Optional[int],
        dataset_info: Optional[Dict[str, Union[str, int, bool, Dict]]],
    ) -> List[Dict[str, Any]]:
        """Sync navigation index from UI to backend state.

        This callback is purely for backend synchronization - it doesn't render UI.
        It listens to changes in the navigation index and updates backend accordingly.
        """
        # Input validations
        assert datapoint_idx is None or isinstance(
            datapoint_idx, int
        ), f"datapoint_idx must be int or None, got {type(datapoint_idx)}"
        assert dataset_info is None or isinstance(
            dataset_info, dict
        ), f"dataset_info must be dict or None, got {type(dataset_info)}"
        assert (
            dataset_info is None or dataset_info == {} or ('name' in dataset_info)
        ), "dataset_info must include name"

        # Handle case where no dataset is selected (normal UI state)
        if datapoint_idx is None or dataset_info is None or dataset_info == {}:
            # Thread-safe return instead of raising PreventUpdate in debounced context
            return [dash.no_update]

        logger.info(f"Syncing navigation index to backend: {datapoint_idx}")

        # Update backend state with current index
        viewer.backend.update_state(current_index=datapoint_idx)

        logger.info("Navigation index synced to backend successfully")

        # Return minimal sync signal (not used for UI)
        return [{'synced': True, 'index': datapoint_idx}]
