from metrics.wrappers.hybrid_metric import HybridMetric
from metrics.vision_3d.point_cloud_registration import IsotropicTransformError
from metrics.vision_3d.point_cloud_registration import TransformInlierRatio


metric_cfg = {
    'class': HybridMetric,
    'args': {
        'metrics_cfg': [{
            'class': IsotropicTransformError,
            'args': {
                'use_buffer': False,
            },
        }, {
            'class': TransformInlierRatio,
            'args': {
                'threshold': 0.3,
                'use_buffer': False,
            },
        }],
    },
}
