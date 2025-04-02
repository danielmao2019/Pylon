from criteria.vision_3d.point_cloud_registration import GeoTransformerCriterion


criterion_cfg = {
    'class': GeoTransformerCriterion,
    'args': {
        'loss': {
            'weight_coarse_loss': 1.0,
            'weight_fine_loss': 1.0,
        },
        'coarse_loss': {
            'positive_margin': 0.1,
            'negative_margin': 1.4,
            'positive_optimal': 0.1,
            'negative_optimal': 1.4,
            'log_scale': 24,
            'positive_overlap': 0.1,
        },
        'fine_loss': {
            'positive_radius': 0.05,
        },
    },
}
