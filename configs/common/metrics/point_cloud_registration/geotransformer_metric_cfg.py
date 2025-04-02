from metrics.vision_3d.point_cloud_registration import GeoTransformerMetric


metric_cfg = {
    'class': GeoTransformerMetric,
    'args': {
        'acceptance_overlap': 0.0,
        'acceptance_radius': 0.1,
        'inlier_ratio_threshold': 0.05,
        'rmse_threshold': 0.2,
        'rre_threshold': 15.0,
        'rte_threshold': 0.3,
    },
}
