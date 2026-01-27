"""3D settings-related callbacks for the viewer."""

from typing import TYPE_CHECKING, Dict, List, Optional

import dash
from dash.dependencies import Input, Output

if TYPE_CHECKING:
    from data.viewer.viewer import DatasetViewer


def register_three_d_settings_density_controls_callbacks(
    app: dash.Dash, viewer: "DatasetViewer"
) -> None:
    # Input validations
    assert isinstance(app, dash.Dash), f"app must be Dash, got {type(app)}"
    assert hasattr(viewer, "backend"), f"viewer must expose backend, got {type(viewer)}"

    @app.callback(
        [Output('density-controls', 'style')],
        [Input('lod-type-dropdown', 'value')],
    )
    def update_density_controls_visibility(
        lod_type: Optional[str],
    ) -> List[Dict[str, str]]:
        """Show density controls only when LOD type is 'none'."""
        # Input validations
        assert lod_type is None or isinstance(
            lod_type, str
        ), f"lod_type must be str or None, got {type(lod_type)}"

        if lod_type == 'none':
            return [{'display': 'block', 'margin-top': '20px'}]
        else:
            return [{'display': 'none'}]
