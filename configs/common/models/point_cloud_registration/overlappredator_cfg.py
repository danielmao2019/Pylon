from models.point_cloud_registration.overlappredator import OverlapPredator


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
        'first_subsampling_dl': 0.3,
        'conv_radius': 4.25,
        'in_feats_dim': 1,
        'first_feats_dim': 256,
        'num_kernel_points': 15,
        'final_feats_dim': 32,
        'condition_feature': True,
        'add_cross_score': True,
        'gnn_feats_dim': 256,
        'dgcnn_k': 10,
        'num_head': 4,
        'nets': ['self', 'cross', 'self'],
    },
}
