"""
DATA.STRUCTURES.THREE_D.POINT_CLOUD.OPS.RENDERING API
"""

from data.structures.three_d.point_cloud.ops.rendering.common import (
    apply_point_size_postprocessing,
    prepare_points_for_rendering,
    validate_rendering_inputs,
)
from data.structures.three_d.point_cloud.ops.rendering.render_depth import (
    render_depth_from_point_cloud,
    render_depth_from_rendering_points,
)
from data.structures.three_d.point_cloud.ops.rendering.render_mask import (
    render_mask_from_rendering_points,
)
from data.structures.three_d.point_cloud.ops.rendering.render_normal import (
    render_normal_from_point_cloud_2d,
    render_normal_from_point_cloud_3d,
    render_normal_from_rendering_points_3d,
)
from data.structures.three_d.point_cloud.ops.rendering.render_rgb import (
    render_rgb_from_point_cloud,
    render_rgb_from_rendering_points,
)
from data.structures.three_d.point_cloud.ops.rendering.render_segmentation import (
    render_segmentation_from_point_cloud,
    render_segmentation_from_rendering_points,
)

__all__ = (
    'apply_point_size_postprocessing',
    'prepare_points_for_rendering',
    'validate_rendering_inputs',
    'render_depth_from_point_cloud',
    'render_depth_from_rendering_points',
    'render_mask_from_rendering_points',
    'render_normal_from_point_cloud_2d',
    'render_normal_from_point_cloud_3d',
    'render_normal_from_rendering_points_3d',
    'render_rgb_from_point_cloud',
    'render_rgb_from_rendering_points',
    'render_segmentation_from_point_cloud',
    'render_segmentation_from_rendering_points',
)
