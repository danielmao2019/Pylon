from criteria.vision_3d.point_cloud_registration.overlappredator_criterion import OverlapPredatorCriterion


overlappredator_criterion_cfg = {
    'class': OverlapPredatorCriterion,
    'args': {
        'log_scale': 48,
        'pos_optimal': 0.1,
        'neg_optimal': 1.4,
        'pos_margin': 0.1,
        'neg_margin': 1.4,
        'max_points': 512,
        'safe_radius': 0.75,
        'matchability_radius': 0.3,
        'pos_radius': 0.21,
        'w_circle_loss': 1.0,
        'w_overlap_loss': 1.0,
        'w_saliency_loss': 0.0,
    },
}
