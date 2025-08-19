from metrics.vision_3d.point_cloud_registration import PARENetMetric


metric_cfg = {
    'class': PARENetMetric,
    'args': {
        # Inlier ratio parameters
        'inlier_threshold': 0.1,
        
        # Evaluation parameters (for PARENet-specific metrics)
        'acceptance_overlap': 0.1,
        'acceptance_radius': 0.1,
        'rmse_threshold': 0.2,
        'feat_rre_threshold': 30.0,
    },
}
