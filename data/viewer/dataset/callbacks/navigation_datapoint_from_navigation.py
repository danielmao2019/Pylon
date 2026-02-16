"""Navigation-related callbacks for the viewer."""

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import dash
from dash import html
from dash.dependencies import Input, Output, State

from data.viewer.dataset.callbacks.navigation_helpers import _build_display_content
from data.viewer.utils.debounce import debounce

if TYPE_CHECKING:
    from data.viewer.dataset.viewer import DatasetViewer

logger = logging.getLogger(__name__)


def register_navigation_datapoint_from_navigation_callbacks(
    app: dash.Dash, viewer: "DatasetViewer"
) -> None:
    # Input validations
    assert isinstance(app, dash.Dash), f"app must be Dash, got {type(app)}"
    assert hasattr(viewer, "backend"), f"viewer must expose backend, got {type(viewer)}"

    @app.callback(
        [Output("datapoint-display", "children", allow_duplicate=True)],
        [Input("datapoint-index-slider", "value")],
        [
            State("dataset-info", "data"),
            State("transforms-store", "data"),
            State("3d-settings-store", "data"),
            State("camera-state", "data"),
        ],
    )
    @debounce
    def update_datapoint_from_navigation(
        datapoint_idx: int,
        dataset_info: Optional[Dict[str, Union[str, int, bool, Dict]]],
        transforms_store: Optional[List[int]],
        settings_3d: Optional[Dict[str, Union[str, int, float, bool]]],
        camera_state: Optional[Dict[str, Any]],
    ) -> List[html.Div]:
        # Input validations
        assert isinstance(
            datapoint_idx, int
        ), f"datapoint_idx must be int, got {type(datapoint_idx)}"
        assert dataset_info is None or isinstance(
            dataset_info, dict
        ), f"dataset_info must be dict or None, got {type(dataset_info)}"
        assert transforms_store is None or isinstance(
            transforms_store, list
        ), f"transforms_store must be list or None, got {type(transforms_store)}"
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
        assert (
            dataset_info is None or dataset_info == {} or ('transforms' in dataset_info)
        ), "dataset_info must include transforms"
        assert (
            dataset_info is None or dataset_info == {} or (transforms_store is not None)
        ), "transforms_store must be initialized before navigation"

        logger.info(f"Navigation callback triggered - Index: {datapoint_idx}")
        if dataset_info is None or dataset_info == {}:
            return [dash.no_update]

        final_camera_state = camera_state or {}
        display = _build_display_content(
            datapoint_idx=datapoint_idx,
            dataset_info=dataset_info,
            transforms_store=transforms_store,
            settings_3d=settings_3d,
            camera_state=final_camera_state,
            viewer=viewer,
        )
        logger.info("Navigation display created successfully")
        return [display]
