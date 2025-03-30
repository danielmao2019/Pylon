from metrics.vision_3d.point_cloud_registration import OverlapPredatorMetric


metric_cfg = {
    'class': OverlapPredatorMetric,
    'args': {
        'max_points': 512,
        'matchability_radius': 0.3,
        'pos_radius': 0.21,
    },
}
