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

from data.viewer.utils.debounce import debounce
from data.viewer.utils.settings_config import ViewerSettings

if TYPE_CHECKING:
    from data.viewer.dataset.viewer import DatasetViewer

logger = logging.getLogger(__name__)


def register_backend_sync_3d_settings_callbacks(
    app: dash.Dash, viewer: "DatasetViewer"
) -> None:
    # Input validations
    assert isinstance(app, dash.Dash), f"app must be Dash, got {type(app)}"
    assert hasattr(viewer, "backend"), f"viewer must expose backend, got {type(viewer)}"

    @app.callback(
        [Output('backend-sync-3d-settings', 'data')],
        [Input('3d-settings-store', 'data')],
    )
    @debounce
    def sync_3d_settings_to_backend(
        settings_3d: Optional[Dict[str, Union[str, int, float, bool]]],
    ) -> List[Dict[str, Any]]:
        """Sync 3D settings from UI store to backend state.

        This callback is purely for backend synchronization - it doesn't render UI.
        It listens to changes in the 3D settings store and updates backend accordingly.
        """
        # Input validations
        assert settings_3d is None or isinstance(
            settings_3d, dict
        ), f"settings_3d must be dict or None, got {type(settings_3d)}"

        if not settings_3d:
            # Thread-safe return instead of raising PreventUpdate in debounced context
            return [dash.no_update]

        logger.info(f"Syncing 3D settings to backend: {settings_3d}")

        # Validate and apply defaults using centralized configuration
        validated_settings = ViewerSettings.validate_3d_settings(
            ViewerSettings.get_3d_settings_with_defaults(settings_3d)
        )

        # Update backend state with validated settings
        viewer.backend.update_state(**validated_settings)

        logger.info("3D settings synced to backend successfully")

        # Return minimal sync signal (not used for UI)
        return [{'synced': True, 'timestamp': str(id(validated_settings))}]
