"""D3Feat metric configuration for Pylon."""

from metrics.vision_3d.point_cloud_registration import D3FeatDescriptorMetric

metric_cfg = {
    'class': D3FeatDescriptorMetric,
    'args': {
        'distance_threshold': 0.1,
        'use_buffer': True
    }
}