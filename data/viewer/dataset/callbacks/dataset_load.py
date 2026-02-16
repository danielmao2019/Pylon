"""Dataset-related callbacks for the viewer."""

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import dash
from dash import html
from dash.dependencies import Input, Output

from data.viewer.dataset.layout.controls.transforms import create_transforms_section
from data.viewer.dataset.layout.display.dataset import create_dataset_info_display

if TYPE_CHECKING:
    from data.viewer.dataset.viewer import DatasetViewer

logger = logging.getLogger(__name__)


# =========================================================================
# Callback Registration
# =========================================================================


def register_dataset_load_callbacks(app: dash.Dash, viewer: "DatasetViewer") -> None:
    # Input validations
    assert isinstance(app, dash.Dash), f"app must be Dash, got {type(app)}"
    assert hasattr(viewer, "backend"), f"viewer must expose backend, got {type(viewer)}"

    # =====================================================================
    # Dataset Loading and UI Management
    # =====================================================================

    @app.callback(
        [
            Output('dataset-info', 'data'),
            Output('datapoint-index-slider', 'min'),
            Output('datapoint-index-slider', 'max'),
            Output('datapoint-index-slider', 'value'),
            Output('datapoint-index-slider', 'marks'),
            Output('datapoint-display', 'children', allow_duplicate=True),
            Output('dataset-info-display', 'children'),
            Output('transforms-section', 'children'),
            Output('transforms-store', 'data', allow_duplicate=True),
        ],
        Input('dataset-dropdown', 'value'),
    )
    def load_dataset(
        dataset_key: Optional[str],
    ) -> List[Union[Dict[str, Any], int, html.Div, List]]:
        """Load a selected dataset and update all related UI components.

        This is a PURE UI callback - it only updates UI components.
        Backend synchronization happens in backend_sync.py automatically.
        """
        # Input validations
        assert dataset_key is None or isinstance(
            dataset_key, str
        ), f"dataset_key must be str or None, got {type(dataset_key)}"

        logger.info(f"Dataset loading callback triggered with dataset: {dataset_key}")

        if dataset_key is None:
            logger.info("No dataset selected")
            return _create_empty_dataset_state()

        # Load dataset using backend (read-only operation)
        logger.info(f"Attempting to load dataset: {dataset_key}")
        dataset_info = viewer.backend.load_dataset(dataset_key)

        # Create slider marks for navigation
        marks = _create_slider_marks(dataset_info['length'])

        # Create success message
        success_message = html.Div(
            f"Dataset '{dataset_key}' loaded successfully with {dataset_info['length']} datapoints. "
            "Use the slider to navigate."
        )

        logger.info("Dataset loaded successfully, returning updated UI components")

        # Initialize all transform indices as selected by default
        all_transform_indices = [
            transform['index'] for transform in dataset_info['transforms']
        ]

        return [
            dataset_info,  # dataset-info
            0,  # slider min
            dataset_info['length'] - 1,  # slider max
            0,  # slider value
            marks,  # slider marks
            success_message,  # datapoint-display
            create_dataset_info_display(dataset_info),  # dataset-info-display
            create_transforms_section(dataset_info['transforms']),  # transforms-section
            all_transform_indices,  # transforms-store (initialize with all selected)
        ]


# =========================================================================
# Helper Functions
# =========================================================================


def _create_empty_dataset_state() -> List[Union[Dict[str, Any], int, html.Div, List]]:
    """Create UI state for when no dataset is selected."""
    return [
        {},  # dataset-info (empty)
        0,  # slider min
        0,  # slider max
        0,  # slider value
        {},  # slider marks (empty)
        html.Div("No dataset selected."),  # datapoint-display
        create_dataset_info_display(),  # dataset-info-display
        create_transforms_section(),  # transforms-section
        [],  # transforms-store (empty list)
    ]


def _create_slider_marks(dataset_length: int) -> Dict[int, str]:
    """Create slider marks based on dataset length."""
    # Input validations
    assert isinstance(
        dataset_length, int
    ), f"dataset_length must be int, got {type(dataset_length)}"
    assert (
        dataset_length >= 0
    ), f"dataset_length must be non-negative, got {dataset_length}"

    if dataset_length <= 10:
        # Show all indices for small datasets
        return {i: str(i) for i in range(dataset_length)}
    else:
        # Show ~10 evenly spaced marks for larger datasets
        step = max(1, dataset_length // 10)
        marks = {i: str(i) for i in range(0, dataset_length, step)}
        # Always include the last index
        marks[dataset_length - 1] = str(dataset_length - 1)
        return marks
