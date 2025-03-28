import models


model_cfg = {
    'class': models.point_cloud_registration.geotransformer.geotransformer.GeoTransformer,
    'args': {
        'cfg': {
            'model': {
                'ground_truth_matching_radius': 0.05,
                'num_points_in_patch': 64,
                'num_sinkhorn_iterations': 100
            },
            'backbone': {
                'num_stages': 4,
                'init_voxel_size': 0.025,
                'kernel_size': 15,
                'base_radius': 2.5,
                'base_sigma': 2.0,
                'init_radius': 0.0625,  # base_radius * init_voxel_size
                'init_sigma': 0.05,     # base_sigma * init_voxel_size
                'group_norm': 32,
                'input_dim': 1,
                'init_dim': 64,
                'output_dim': 256
            },
            'geotransformer': {
                'input_dim': 1024,
                'hidden_dim': 256,
                'output_dim': 256,
                'num_heads': 4,
                'blocks': ['self', 'cross', 'self', 'cross', 'self', 'cross'],
                'sigma_d': 0.2,
                'sigma_a': 15,
                'angle_k': 3,
                'reduction_a': 'max'
            },
            'coarse_matching': {
                'num_targets': 128,
                'overlap_threshold': 0.1,
                'num_correspondences': 256,
                'dual_normalization': True
            },
            'fine_matching': {
                'topk': 3,
                'acceptance_radius': 0.1,
                'mutual': True,
                'confidence_threshold': 0.05,
                'use_dustbin': False,
                'use_global_score': False,
                'correspondence_threshold': 3,
                'correspondence_limit': None,
                'num_refinement_steps': 5
            },
        },
    },
}
