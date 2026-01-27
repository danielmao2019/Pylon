"""Dataset-related callbacks for the viewer."""

from typing import TYPE_CHECKING, Dict, List, Optional

import dash
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate

from data.viewer.callbacks.dataset import TYPE_LABELS

if TYPE_CHECKING:
    from data.viewer.viewer import DatasetViewer


# =========================================================================
# Callback Registration
# =========================================================================


def register_dataset_group_reload_callbacks(
    app: dash.Dash, viewer: "DatasetViewer"
) -> None:
    # Input validations
    assert isinstance(app, dash.Dash), f"app must be Dash, got {type(app)}"
    assert hasattr(viewer, "backend"), f"viewer must expose backend, got {type(viewer)}"

    # =====================================================================
    # Dataset Group Management (Hierarchical Dropdown - Top Level)
    # =====================================================================

    @app.callback(
        Output('dataset-group-dropdown', 'options'),
        Input('reload-button', 'n_clicks'),
    )
    def reload_dataset_groups(n_clicks: Optional[int]) -> List[Dict[str, str]]:
        """Reload available dataset groups for hierarchical dropdown."""
        # Input validations
        assert n_clicks is None or isinstance(
            n_clicks, int
        ), f"n_clicks must be int or None, got {type(n_clicks)}"

        if n_clicks is None:
            raise PreventUpdate

        hierarchical_datasets = viewer.backend.get_available_datasets_hierarchical()

        # Create group dropdown options
        options = []
        for dataset_type in sorted(hierarchical_datasets.keys()):
            assert dataset_type in TYPE_LABELS, f"Unknown dataset type: {dataset_type}"
            label = TYPE_LABELS[dataset_type]
            dataset_count = len(hierarchical_datasets[dataset_type])
            options.append(
                {'label': f"{label} ({dataset_count} datasets)", 'value': dataset_type}
            )

        return options
