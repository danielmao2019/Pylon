from models.point_cloud_registration.overlappredator.overlappredator import OverlapPredator


model_cfg = {
    'class': OverlapPredator,
    'args': {
        'architecture': [
            'simple',
            'resnetb',
            'resnetb_strided',
            'resnetb',
            'resnetb',
            'resnetb_strided',
            'resnetb',
            'resnetb',
            'resnetb_strided',
            'resnetb',
            'resnetb',
            'nearest_upsample',
            'unary',
            'nearest_upsample',
            'unary',
            'nearest_upsample',
            'last_unary',
        ],
        'num_layers': 4,
        'in_points_dim': 3,
        'first_feats_dim': 256,
        'final_feats_dim': 32,
        'first_subsampling_dl': 0.3,
        'in_feats_dim': 1,
        'conv_radius': 4.25,
        'deform_radius': 5.0,
        'num_kernel_points': 15,
        'KP_extent': 2.0,
        'KP_influence': 'linear',
        'aggregation_mode': 'sum',
        'fixed_kernel_points': 'center',
        'use_batch_norm': True,
        'batch_norm_momentum': 0.02,
        'deformable': False,
        'modulated': False,
        'add_cross_score': True,
        'condition_feature': True,
        'gnn_feats_dim': 256,
        'dgcnn_k': 10,
        'num_head': 4,
        'nets': ['self', 'cross', 'self'],
    },
}
