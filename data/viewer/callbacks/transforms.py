"""Transform selection callbacks for the viewer."""

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import dash
from dash import html
from dash.dependencies import ALL, Input, Output, State

from data.viewer.utils.debounce import debounce
from data.viewer.utils.settings_config import ViewerSettings

if TYPE_CHECKING:
    from data.viewer.viewer import DatasetViewer

logger = logging.getLogger(__name__)


def register_transforms_callbacks(app: dash.Dash, viewer: "DatasetViewer") -> None:
    # Input validations
    assert isinstance(app, dash.Dash), f"app must be Dash, got {type(app)}"
    assert hasattr(viewer, "backend"), f"viewer must expose backend, got {type(viewer)}"

    @app.callback(
        [Output('datapoint-display', 'children'), Output('transforms-store', 'data')],
        [Input({'type': 'transform-checkbox', 'index': ALL}, 'value')],
        [
            State('dataset-info', 'data'),
            State('datapoint-index-slider', 'value'),
            State('3d-settings-store', 'data'),
            State('camera-state', 'data'),
        ],
    )
    @debounce
    def update_datapoint_from_transforms(
        transform_values: List[List[int]],
        dataset_info: Optional[Dict[str, Union[str, int, bool, Dict]]],
        datapoint_idx: int,
        settings_3d: Optional[Dict[str, Union[str, int, float, bool]]],
        camera_state: Optional[Dict[str, Any]],
    ) -> List[Union[html.Div, Dict]]:
        """
        Update the displayed datapoint when transform selections change.
        Also handles 3D point cloud visualization settings.
        """
        # Input validations
        assert isinstance(
            transform_values, list
        ), f"transform_values must be list, got {type(transform_values)}"
        assert dataset_info is None or isinstance(
            dataset_info, dict
        ), f"dataset_info must be dict or None, got {type(dataset_info)}"
        assert isinstance(
            datapoint_idx, int
        ), f"datapoint_idx must be int, got {type(datapoint_idx)}"
        assert settings_3d is None or isinstance(
            settings_3d, dict
        ), f"settings_3d must be dict or None, got {type(settings_3d)}"
        assert camera_state is None or isinstance(
            camera_state, dict
        ), f"camera_state must be dict or None, got {type(camera_state)}"
        assert (
            dataset_info is None or dataset_info == {} or ('name' in dataset_info)
        ), "dataset_info must include name"
        assert (
            dataset_info is None or dataset_info == {} or ('type' in dataset_info)
        ), "dataset_info must include type"

        logger.info(
            f"Transform selection callback triggered - Transform values: {transform_values}"
        )

        # Handle case where no dataset is selected (normal UI state)
        if dataset_info is None or dataset_info == {}:
            # Thread-safe return instead of raising PreventUpdate in debounced context
            return [dash.no_update, dash.no_update]

        dataset_name: str = dataset_info['name']
        dataset_type: str = dataset_info['type']
        logger.info(f"Updating datapoint for dataset: {dataset_name}")

        # Get list of selected transform indices
        selected_indices = [
            idx
            for values in transform_values
            for idx in values  # values will be a list containing the index if checked, empty if not
        ]

        # Get datapoint from backend using kwargs
        datapoint = viewer.backend.get_datapoint(
            dataset_name=dataset_name,
            index=datapoint_idx,
            transform_indices=selected_indices,
        )

        logger.info(f"Dataset type: {dataset_type}")

        # Extract 3D settings and class labels using centralized configuration
        settings_3d = ViewerSettings.get_3d_settings_with_defaults(settings_3d)

        # Get dataset instance for display method and class labels
        dataset_instance = viewer.backend.get_dataset_instance(
            dataset_name=dataset_name
        )
        class_labels = (
            dataset_instance.class_labels
            if hasattr(dataset_instance, 'class_labels')
            and dataset_instance.class_labels
            else None
        )

        # All datasets must have display_datapoint method from base classes
        assert (
            dataset_instance is not None
        ), f"Dataset instance must not be None for dataset: {dataset_name}"
        assert hasattr(
            dataset_instance, 'display_datapoint'
        ), f"Dataset {type(dataset_instance).__name__} must have display_datapoint method"

        display_func = dataset_instance.display_datapoint
        logger.info(
            f"Using display method from dataset class: {type(dataset_instance).__name__}"
        )

        # Check if camera state is the default state - if so, pass None to allow camera pose calculation
        default_camera_state = {
            'eye': {'x': 1.5, 'y': 1.5, 'z': 1.5},
            'center': {'x': 0, 'y': 0, 'z': 0},
            'up': {'x': 0, 'y': 0, 'z': 1},
        }

        # For datasets that have camera poses (like iVISION MT), pass None to trigger pose calculation
        final_camera_state = camera_state
        if camera_state == default_camera_state:
            # Check if datapoint has camera pose - if so, let dataset calculate camera from pose
            if (
                'meta_info' in datapoint
                and 'camera_pose' in datapoint['meta_info']
                and hasattr(dataset_instance, '__class__')
                and 'iVISION' in dataset_instance.__class__.__name__
            ):
                final_camera_state = None
                logger.info(
                    f"Using None camera_state for {dataset_instance.__class__.__name__} to trigger camera pose calculation"
                )

        # Create display using the determined display function
        logger.info(f"Creating {dataset_type} display with selected transforms")
        display = display_func(
            datapoint=datapoint,
            class_labels=class_labels,
            camera_state=final_camera_state,
            settings_3d=settings_3d,
        )

        logger.info("Display created successfully with transform selection")
        # Store the selected transform indices for use by navigation callback
        return [display, selected_indices]
