from criteria.vision_3d.point_cloud_registration import PARENetCriterion


criterion_cfg = {
    'class': PARENetCriterion,
    'args': {
        # Coarse loss parameters
        'coarse_positive_margin': 0.1,
        'coarse_negative_margin': 1.4,
        'coarse_positive_optimal': 0.1,
        'coarse_negative_optimal': 1.4,
        'coarse_log_scale': 10.0,
        'coarse_positive_overlap': 0.1,

        # Fine loss parameters
        'fine_positive_radius': 0.1,
        'fine_negative_radius': 0.1,
        'fine_positive_margin': 0.05,
        'fine_negative_margin': 0.2,

        # Loss weights
        'weight_coarse_loss': 1.0,
        'weight_fine_ri_loss': 1.0,
        'weight_fine_re_loss': 1.0,
    },
}
