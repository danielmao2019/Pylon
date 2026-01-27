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
from dash.dependencies import Input, Output

if TYPE_CHECKING:
    from data.viewer.viewer import DatasetViewer

logger = logging.getLogger(__name__)


def register_backend_sync_dataset_callbacks(
    app: dash.Dash, viewer: "DatasetViewer"
) -> None:
    # Input validations
    assert isinstance(app, dash.Dash), f"app must be Dash, got {type(app)}"
    assert hasattr(viewer, "backend"), f"viewer must expose backend, got {type(viewer)}"

    @app.callback(
        [Output('backend-sync-dataset', 'data')],
        [Input('dataset-info', 'data')],
    )
    def sync_dataset_to_backend(
        dataset_info: Optional[Dict[str, Union[str, int, bool, Dict]]],
    ) -> List[Dict[str, Any]]:
        """Sync dataset info from UI store to backend state.

        This callback is purely for backend synchronization - it doesn't render UI.
        It listens to changes in dataset info and updates backend accordingly.
        """
        # Input validations
        assert dataset_info is None or isinstance(
            dataset_info, dict
        ), f"dataset_info must be dict or None, got {type(dataset_info)}"
        assert (
            dataset_info is None or dataset_info == {} or ('name' in dataset_info)
        ), "dataset_info must include name"

        # Handle case where no dataset is selected (normal UI state)
        if dataset_info is None or dataset_info == {}:
            # Handle dataset deselection or empty dataset info
            logger.info("No dataset selected - clearing backend state")
            viewer.backend.current_dataset = None
            return [{'synced': True, 'dataset': None}]

        dataset_name: str = dataset_info['name']
        logger.info(f"Syncing dataset to backend: {dataset_name}")

        # Update backend state with dataset info
        viewer.backend.update_state(
            current_dataset=dataset_name,
            current_index=0,  # Reset to first datapoint when dataset changes
        )

        logger.info("Dataset info synced to backend successfully")

        # Return minimal sync signal (not used for UI)
        return [{'synced': True, 'dataset': dataset_name}]
