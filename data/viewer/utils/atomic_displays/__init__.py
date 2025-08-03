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

# Import all atomic display functions for easy access
from data.viewer.utils.atomic_displays.image_display import (
    create_image_display, 
    get_image_display_stats,
    create_image_figure,
    get_image_stats,
    image_to_numpy
)
from data.viewer.utils.atomic_displays.depth_display import (
    create_depth_display,
    get_depth_display_stats
)
from data.viewer.utils.atomic_displays.normal_display import (
    create_normal_display,
    get_normal_display_stats
)
from data.viewer.utils.atomic_displays.edge_display import (
    create_edge_display,
    get_edge_display_stats
)
from data.viewer.utils.atomic_displays.segmentation_display import (
    create_segmentation_display,
    get_segmentation_display_stats
)
from data.viewer.utils.atomic_displays.point_cloud_display import (
    create_point_cloud_display,
    get_point_cloud_display_stats,
    build_point_cloud_display_id,
    create_point_cloud_figure,
    get_point_cloud_stats,
    build_point_cloud_id,
    apply_lod_to_point_cloud,
    normalize_point_cloud_id,
    point_cloud_to_numpy
)
from data.viewer.utils.atomic_displays.instance_surrogate_display import (
    create_instance_surrogate_display,
    get_instance_surrogate_display_stats
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
    # Stats functions
    'get_image_display_stats',
    'get_depth_display_stats',
    'get_normal_display_stats', 
    'get_edge_display_stats',
    'get_segmentation_display_stats',
    'get_point_cloud_display_stats',
    'get_instance_surrogate_display_stats',
    # Core image functions (now available)
    'create_image_figure',
    'get_image_stats',
    'image_to_numpy',
    # Core point cloud functions (now available)
    'create_point_cloud_figure',
    'get_point_cloud_stats',
    'build_point_cloud_id',
    'apply_lod_to_point_cloud',
    'normalize_point_cloud_id',
    'point_cloud_to_numpy',
    # Utility functions
    'build_point_cloud_display_id'
]