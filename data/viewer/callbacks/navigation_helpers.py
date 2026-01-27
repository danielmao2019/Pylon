"""Navigation-related callbacks for the viewer."""

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from dash import html

from data.viewer.utils.settings_config import ViewerSettings

if TYPE_CHECKING:
    from data.viewer.viewer import DatasetViewer

logger = logging.getLogger(__name__)


def _build_display_content(
    datapoint_idx: int,
    dataset_info: Dict[str, Union[str, int, bool, Dict]],
    transforms_store: List[int],
    settings_3d: Optional[Dict[str, Union[str, int, float, bool]]],
    camera_state: Dict[str, Any],
    viewer: "DatasetViewer",
) -> html.Div:
    # Input validations
    assert isinstance(
        datapoint_idx, int
    ), f"datapoint_idx must be int, got {type(datapoint_idx)}"
    assert isinstance(
        dataset_info, dict
    ), f"dataset_info must be dict, got {type(dataset_info)}"
    assert 'name' in dataset_info, "dataset_info must include name"
    assert 'type' in dataset_info, "dataset_info must include type"
    assert isinstance(
        transforms_store, list
    ), f"transforms_store must be list, got {type(transforms_store)}"
    assert settings_3d is None or isinstance(
        settings_3d, dict
    ), f"settings_3d must be dict or None, got {type(settings_3d)}"
    assert isinstance(
        camera_state, dict
    ), f"camera_state must be dict, got {type(camera_state)}"
    assert hasattr(viewer, "backend"), f"viewer must expose backend, got {type(viewer)}"

    dataset_name: str = dataset_info["name"]
    dataset_type: str = dataset_info["type"]
    logger.info(f"Navigating to index {datapoint_idx} in dataset: {dataset_name}")

    datapoint = viewer.backend.get_datapoint(
        dataset_name=dataset_name,
        index=datapoint_idx,
        transform_indices=transforms_store,
    )

    logger.info(f"Dataset type: {dataset_type}")

    resolved_settings = ViewerSettings.get_3d_settings_with_defaults(settings_3d)
    dataset_instance = viewer.backend.get_dataset_instance(dataset_name=dataset_name)
    class_labels = (
        dataset_instance.class_labels
        if hasattr(dataset_instance, "class_labels") and dataset_instance.class_labels
        else None
    )

    assert (
        dataset_instance is not None
    ), f"Dataset instance must not be None for dataset: {dataset_name}"
    assert hasattr(
        dataset_instance, "display_datapoint"
    ), f"Dataset {type(dataset_instance).__name__} must have display_datapoint method"

    default_camera_state = {
        "eye": {"x": 1.5, "y": 1.5, "z": 1.5},
        "center": {"x": 0, "y": 0, "z": 0},
        "up": {"x": 0, "y": 0, "z": 1},
    }

    final_camera_state = camera_state
    if camera_state == default_camera_state:
        if (
            "meta_info" in datapoint
            and "camera_pose" in datapoint["meta_info"]
            and "camera_intrinsics" in datapoint["meta_info"]
        ):
            final_camera_state = None
            logger.info(
                f"Using None camera_state for {dataset_instance.__class__.__name__} to trigger camera pose calculation"
            )

    display = dataset_instance.display_datapoint(
        datapoint=datapoint,
        class_labels=class_labels,
        camera_state=final_camera_state,
        settings_3d=resolved_settings,
    )
    return display
