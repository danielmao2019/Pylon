"""Dataset-related callbacks for the viewer."""

from typing import TYPE_CHECKING, Dict, List, Optional, Union

import dash
from dash.dependencies import Input, Output

if TYPE_CHECKING:
    from data.viewer.viewer import DatasetViewer


# =========================================================================
# Callback Registration
# =========================================================================


def register_dataset_options_callbacks(
    app: dash.Dash, viewer: "DatasetViewer"
) -> None:
    # Input validations
    assert isinstance(app, dash.Dash), f"app must be Dash, got {type(app)}"
    assert hasattr(viewer, "backend"), f"viewer must expose backend, got {type(viewer)}"

    # =====================================================================
    # Dataset Selection Management (Hierarchical Dropdown - Second Level)
    # =====================================================================

    @app.callback(
        [
            Output('dataset-dropdown', 'options'),
            Output('dataset-dropdown', 'disabled'),
            Output('dataset-dropdown', 'placeholder'),
            Output('dataset-dropdown', 'value'),
        ],
        Input('dataset-group-dropdown', 'value'),
    )
    def update_dataset_options(
        selected_group: Optional[str],
    ) -> List[Union[List[Dict[str, str]], bool, str, None]]:
        """Update dataset dropdown options based on selected group."""
        # Input validations
        assert selected_group is None or isinstance(
            selected_group, str
        ), f"selected_group must be str or None, got {type(selected_group)}"

        if selected_group is None:
            return [[], True, "First select a category above...", None]

        hierarchical_datasets = viewer.backend.get_available_datasets_hierarchical()
        assert (
            selected_group in hierarchical_datasets
        ), f"Unknown dataset group: {selected_group}"

        # Create options for the selected group
        options = []
        for dataset_key, dataset_name in sorted(
            hierarchical_datasets[selected_group].items()
        ):
            options.append({'label': dataset_name, 'value': dataset_key})

        placeholder = f"Choose from {len(options)} available datasets..."
        return [options, False, placeholder, None]
