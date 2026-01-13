"""Point cloud rendering utilities."""

# Main rendering functions (high-level API)
from data.structures.three_d.point_cloud.ops.rendering.render_rgb import render_rgb_from_point_cloud
from data.structures.three_d.point_cloud.ops.rendering.render_depth import render_depth_from_point_cloud
from data.structures.three_d.point_cloud.ops.rendering.render_segmentation import (
    render_segmentation_from_point_cloud,
)
from data.structures.three_d.point_cloud.ops.rendering.render_normal import (
    render_normal_from_point_cloud_2d,
    render_normal_from_point_cloud_3d,
)

# Helper rendering functions (low-level API)
from data.structures.three_d.point_cloud.ops.rendering.render_rgb import render_rgb_from_rendering_points
from data.structures.three_d.point_cloud.ops.rendering.render_depth import (
    render_depth_from_rendering_points,
)
from data.structures.three_d.point_cloud.ops.rendering.render_segmentation import (
    render_segmentation_from_rendering_points,
)
from data.structures.three_d.point_cloud.ops.rendering.render_normal import (
    render_normal_from_rendering_points_3d,
)
from data.structures.three_d.point_cloud.ops.rendering.render_mask import (
    render_mask_from_rendering_points,
)

# Common utilities
from data.structures.three_d.point_cloud.ops.rendering.common import (
    validate_rendering_inputs,
    prepare_points_for_rendering,
    apply_point_size_postprocessing,
)


__all__ = [
    # Main rendering functions
    'render_rgb_from_point_cloud',
    'render_depth_from_point_cloud',
    'render_segmentation_from_point_cloud',
    'render_normal_from_point_cloud_2d',
    'render_normal_from_point_cloud_3d',
    # Helper rendering functions
    'render_rgb_from_rendering_points',
    'render_depth_from_rendering_points',
    'render_segmentation_from_rendering_points',
    'render_normal_from_rendering_points_3d',
    'render_mask_from_rendering_points',
    # Common utilities
    'validate_rendering_inputs',
    'prepare_points_for_rendering',
    'apply_point_size_postprocessing',
]
