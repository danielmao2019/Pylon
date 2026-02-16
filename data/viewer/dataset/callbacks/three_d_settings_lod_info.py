"""3D settings-related callbacks for the viewer."""

from typing import TYPE_CHECKING, List, Optional

import dash
from dash.dependencies import Input, Output

if TYPE_CHECKING:
    from data.viewer.dataset.viewer import DatasetViewer


def register_three_d_settings_lod_info_callbacks(
    app: dash.Dash, viewer: "DatasetViewer"
) -> None:
    # Input validations
    assert isinstance(app, dash.Dash), f"app must be Dash, got {type(app)}"
    assert hasattr(viewer, "backend"), f"viewer must expose backend, got {type(viewer)}"

    @app.callback(
        [Output('lod-info-display', 'children')],
        [Input('lod-type-dropdown', 'value')],
    )
    def update_lod_info_display(lod_type: Optional[str]) -> List[str]:
        """Update LOD information display based on selected LOD type."""
        # Input validations
        assert lod_type is None or isinstance(
            lod_type, str
        ), f"lod_type must be str or None, got {type(lod_type)}"

        if not lod_type:
            return [""]

        lod_descriptions = {
            'continuous': 'Real-time adaptive sampling based on camera distance. Provides smooth performance scaling.',
            'discrete': 'Fixed LOD levels with 2x downsampling per level. Predictable performance.',
            'none': 'No level of detail - shows all points. Use density control to adjust point count.',
        }
        assert lod_type in lod_descriptions, f"Unknown LOD type: {lod_type}"

        return [lod_descriptions[lod_type]]
