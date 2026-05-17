"""Atomic display utilities for multi-task dataset visualization.

This module provides composable display functions that can be used by multi-task
datasets to implement their display_datapoint methods. Each function handles
a specific data type (images, depth maps, normals, etc.) and follows Pylon's
fail-fast philosophy with comprehensive input validation.

Usage:
    from data.viewer.utils.atomic_displays import (
        create_image_display,
        create_depth_display,
        create_normal_display,
        create_segmentation_display,
        create_edge_display,
        create_point_cloud_display,
        create_instance_surrogate_display
    )
"""

from data.viewer.utils.atomic_displays.mesh.dash.core_mesh_display import (
    create_mesh_display,
)
from data.viewer.utils.atomic_displays.pixels.dash.apis import (
    create_depth_display,
    create_edge_display,
    create_image_display,
    create_instance_surrogate_display,
    create_normal_display,
    create_segmentation_display,
    get_depth_display_stats,
    get_edge_display_stats,
    get_image_display_stats,
    get_instance_surrogate_display_stats,
    get_normal_display_stats,
    get_segmentation_display_stats,
)
from data.viewer.utils.atomic_displays.points.dash.core_points_display import (
    build_point_cloud_id,
    create_point_cloud_display,
    get_point_cloud_display_stats,
    normalize_point_cloud_id,
)

__all__ = [
    # Display functions
    'create_image_display',
    'create_depth_display',
    'create_normal_display',
    'create_edge_display',
    'create_segmentation_display',
    'create_point_cloud_display',
    'create_instance_surrogate_display',
    'create_mesh_display',
    # Stats functions
    'get_image_display_stats',
    'get_depth_display_stats',
    'get_normal_display_stats',
    'get_edge_display_stats',
    'get_segmentation_display_stats',
    'get_point_cloud_display_stats',
    'get_instance_surrogate_display_stats',
    # Utility functions
    'build_point_cloud_id',
    'normalize_point_cloud_id',
]
