"""Register dataset viewer callbacks."""

from typing import TYPE_CHECKING

import dash

from data.viewer.dataset.callbacks.backend_sync_3d_settings import (
    register_backend_sync_3d_settings_callbacks,
)
from data.viewer.dataset.callbacks.backend_sync_dataset import (
    register_backend_sync_dataset_callbacks,
)
from data.viewer.dataset.callbacks.backend_sync_navigation import (
    register_backend_sync_navigation_callbacks,
)
from data.viewer.dataset.callbacks.camera_reset import register_camera_reset_callbacks
from data.viewer.dataset.callbacks.camera_sync import register_camera_sync_callbacks
from data.viewer.dataset.callbacks.class_distribution import (
    register_class_distribution_callbacks,
)
from data.viewer.dataset.callbacks.dataset_group_reload import (
    register_dataset_group_reload_callbacks,
)
from data.viewer.dataset.callbacks.dataset_load import register_dataset_load_callbacks
from data.viewer.dataset.callbacks.dataset_options import register_dataset_options_callbacks
from data.viewer.dataset.callbacks.navigation_current_index import (
    register_navigation_current_index_callbacks,
)
from data.viewer.dataset.callbacks.navigation_datapoint_from_camera import (
    register_navigation_datapoint_from_camera_callbacks,
)
from data.viewer.dataset.callbacks.navigation_datapoint_from_navigation import (
    register_navigation_datapoint_from_navigation_callbacks,
)
from data.viewer.dataset.callbacks.navigation_datapoint_from_settings import (
    register_navigation_datapoint_from_settings_callbacks,
)
from data.viewer.dataset.callbacks.navigation_next import register_navigation_next_callbacks
from data.viewer.dataset.callbacks.navigation_prev import register_navigation_prev_callbacks
from data.viewer.dataset.callbacks.three_d_settings_density import (
    register_three_d_settings_density_callbacks,
)
from data.viewer.dataset.callbacks.three_d_settings_density_controls import (
    register_three_d_settings_density_controls_callbacks,
)
from data.viewer.dataset.callbacks.three_d_settings_lod_info import (
    register_three_d_settings_lod_info_callbacks,
)
from data.viewer.dataset.callbacks.three_d_settings_lod_type import (
    register_three_d_settings_lod_type_callbacks,
)
from data.viewer.dataset.callbacks.three_d_settings_point_opacity import (
    register_three_d_settings_point_opacity_callbacks,
)
from data.viewer.dataset.callbacks.three_d_settings_point_size import (
    register_three_d_settings_point_size_callbacks,
)
from data.viewer.dataset.callbacks.three_d_settings_radius import (
    register_three_d_settings_radius_callbacks,
)
from data.viewer.dataset.callbacks.three_d_settings_view_controls import (
    register_three_d_settings_view_controls_callbacks,
)
from data.viewer.dataset.callbacks.transforms import register_transforms_callbacks
from data.viewer.dataset.callbacks.transforms_section import (
    register_transforms_section_callbacks,
)

if TYPE_CHECKING:
    from data.viewer.dataset.viewer import DatasetViewer


def register_viewer_callbacks(app: dash.Dash, viewer: "DatasetViewer") -> None:
    # Input validations
    assert isinstance(app, dash.Dash), f"app must be Dash, got {type(app)}"
    assert hasattr(viewer, "backend"), f"viewer must expose backend, got {type(viewer)}"
    assert hasattr(
        viewer, "available_datasets"
    ), f"viewer must expose available_datasets, got {type(viewer)}"

    register_dataset_group_reload_callbacks(app=app, viewer=viewer)
    register_dataset_options_callbacks(app=app, viewer=viewer)
    register_dataset_load_callbacks(app=app, viewer=viewer)
    register_transforms_section_callbacks(app=app, viewer=viewer)
    register_transforms_callbacks(app=app, viewer=viewer)
    register_three_d_settings_point_size_callbacks(app=app, viewer=viewer)
    register_three_d_settings_point_opacity_callbacks(app=app, viewer=viewer)
    register_three_d_settings_radius_callbacks(app=app, viewer=viewer)
    register_three_d_settings_lod_type_callbacks(app=app, viewer=viewer)
    register_three_d_settings_density_callbacks(app=app, viewer=viewer)
    register_three_d_settings_view_controls_callbacks(app=app, viewer=viewer)
    register_three_d_settings_lod_info_callbacks(app=app, viewer=viewer)
    register_three_d_settings_density_controls_callbacks(app=app, viewer=viewer)
    register_backend_sync_3d_settings_callbacks(app=app, viewer=viewer)
    register_backend_sync_dataset_callbacks(app=app, viewer=viewer)
    register_backend_sync_navigation_callbacks(app=app, viewer=viewer)
    register_navigation_prev_callbacks(app=app, viewer=viewer)
    register_navigation_next_callbacks(app=app, viewer=viewer)
    register_navigation_current_index_callbacks(app=app, viewer=viewer)
    register_navigation_datapoint_from_navigation_callbacks(app=app, viewer=viewer)
    register_navigation_datapoint_from_settings_callbacks(app=app, viewer=viewer)
    register_navigation_datapoint_from_camera_callbacks(app=app, viewer=viewer)
    register_camera_sync_callbacks(app=app, viewer=viewer)
    register_camera_reset_callbacks(app=app, viewer=viewer)
    register_class_distribution_callbacks(app=app, viewer=viewer)
