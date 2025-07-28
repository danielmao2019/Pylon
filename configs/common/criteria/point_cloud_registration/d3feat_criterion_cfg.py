"""D3Feat criterion configuration for Pylon."""

from criteria.vision_3d.point_cloud_registration import CircleLoss

criterion_cfg = {
    'class': CircleLoss,
    'args': {
        'pos_margin': 0.1,
        'neg_margin': 1.4,
        'pos_optimal': 0.1,
        'neg_optimal': 1.4,
        'log_scale': 10.0,
        'safe_radius': 0.04,
        'use_buffer': True
    }
}