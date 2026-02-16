"""Camera pose and synchronization callbacks for the viewer."""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import dash
from dash.dependencies import ALL, Input, Output, State
from dash.exceptions import PreventUpdate

from data.viewer.utils.camera_utils import reset_figure_camera, update_figures_parallel
from data.viewer.utils.settings_config import ViewerSettings

if TYPE_CHECKING:
    from data.viewer.dataset.viewer import DatasetViewer


def register_camera_reset_callbacks(app: dash.Dash, viewer: "DatasetViewer") -> None:
    # Input validations
    assert isinstance(app, dash.Dash), f"app must be Dash, got {type(app)}"
    assert hasattr(viewer, "backend"), f"viewer must expose backend, got {type(viewer)}"

    @app.callback(
        [
            Output(
                {'type': 'point-cloud-graph', 'index': ALL},
                'figure',
                allow_duplicate=True,
            ),
            Output('camera-state', 'data', allow_duplicate=True),
        ],
        [Input('reset-camera-button', 'n_clicks')],
        [State({'type': 'point-cloud-graph', 'index': ALL}, 'figure')],
    )
    def reset_camera_view(
        n_clicks: Optional[int], all_figures: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Reset camera view to Plotly's auto-calculated camera state."""
        # Input validations
        assert n_clicks is None or isinstance(
            n_clicks, int
        ), f"n_clicks must be int or None, got {type(n_clicks)}"
        assert isinstance(
            all_figures, list
        ), f"all_figures must be list, got {type(all_figures)}"

        if n_clicks is None or n_clicks == 0:
            raise PreventUpdate

        # Reset all figures to use Plotly's auto-calculated camera
        reset_func = reset_figure_camera()
        updated_figures = update_figures_parallel(
            all_figures,
            reset_func,
            ViewerSettings.PERFORMANCE_SETTINGS['max_thread_workers'],
        )

        # Return None for camera state since we're using Plotly auto-calculation
        return updated_figures, None
