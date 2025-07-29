"""D3Feat model configuration for Pylon."""

from models.point_cloud_registration.d3feat import D3FeatModel

config = {
    'class': D3FeatModel,
    'args': {
        'num_layers': 5,
        'in_points_dim': 3,
        'first_features_dim': 128,
        'first_subsampling_dl': 0.03,
        'in_features_dim': 1,
        'conv_radius': 2.5,
        'deform_radius': 5.0,
        'num_kernel_points': 15,
        'KP_extent': 2.0,
        'KP_influence': 'linear',
        'aggregation_mode': 'sum',
        'fixed_kernel_points': 'center',
        'use_batch_norm': False,
        'batch_norm_momentum': 0.02,
        'deformable': False,
        'modulated': False,
    }
}