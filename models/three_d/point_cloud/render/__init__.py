"""
MODELS.THREE_D.POINT_CLOUD.RENDER API
"""

from models.three_d.point_cloud.render.display import render_display
from models.three_d.point_cloud.render.common import (
    apply_point_size_postprocessing,
    prepare_points_for_rendering,
    validate_rendering_inputs,
)
from models.three_d.point_cloud.render.render_depth import (
    render_depth_from_point_cloud,
    render_depth_from_rendering_points,
)
from models.three_d.point_cloud.render.render_mask import (
    render_mask_from_rendering_points,
)
from models.three_d.point_cloud.render.render_normal import (
    render_normal_from_point_cloud_2d,
    render_normal_from_point_cloud_3d,
    render_normal_from_rendering_points_3d,
)
from models.three_d.point_cloud.render.render_rgb import (
    render_rgb_from_point_cloud,
    render_rgb_from_rendering_points,
)
from models.three_d.point_cloud.render.render_segmentation import (
    render_segmentation_from_point_cloud,
    render_segmentation_from_rendering_points,
)

__all__ = (
    'render_display',
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
