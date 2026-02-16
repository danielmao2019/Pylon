"""3D settings-related callbacks for the viewer."""

from typing import TYPE_CHECKING, Dict, List, Optional, Union

import dash
from dash.dependencies import Input, Output, State

from data.viewer.dataset.callbacks.three_d_settings_helpers import _build_3d_settings_payload
from data.viewer.utils.debounce import debounce

if TYPE_CHECKING:
    from data.viewer.dataset.viewer import DatasetViewer


def register_three_d_settings_point_opacity_callbacks(
    app: dash.Dash, viewer: "DatasetViewer"
) -> None:
    # Input validations
    assert isinstance(app, dash.Dash), f"app must be Dash, got {type(app)}"
    assert hasattr(viewer, "backend"), f"viewer must expose backend, got {type(viewer)}"

    @app.callback(
        [Output('3d-settings-store', 'data', allow_duplicate=True)],
        [Input('point-opacity-slider', 'value')],
        [
            State('point-size-slider', 'value'),
            State('radius-slider', 'value'),
            State('lod-type-dropdown', 'value'),
            State('density-slider', 'value'),
        ],
    )
    @debounce
    def update_3d_settings_point_opacity(
        point_opacity: Optional[float],
        point_size: Optional[float],
        sym_diff_radius: Optional[float],
        lod_type: Optional[str],
        density_percentage: Optional[int],
    ) -> List[Dict[str, Union[str, int, float, bool]]]:
        """Update 3D settings store when point opacity changes."""
        # Input validations
        assert point_opacity is None or isinstance(
            point_opacity, (int, float)
        ), f"point_opacity must be numeric or None, got {type(point_opacity)}"
        assert point_size is None or isinstance(
            point_size, (int, float)
        ), f"point_size must be numeric or None, got {type(point_size)}"
        assert sym_diff_radius is None or isinstance(
            sym_diff_radius, (int, float)
        ), f"sym_diff_radius must be numeric or None, got {type(sym_diff_radius)}"
        assert lod_type is None or isinstance(
            lod_type, str
        ), f"lod_type must be str or None, got {type(lod_type)}"
        assert density_percentage is None or isinstance(
            density_percentage, int
        ), f"density_percentage must be int or None, got {type(density_percentage)}"

        if point_opacity is None or point_size is None:
            return [dash.no_update]

        return _build_3d_settings_payload(
            point_size=point_size,
            point_opacity=point_opacity,
            sym_diff_radius=sym_diff_radius,
            lod_type=lod_type,
            density_percentage=density_percentage,
        )
