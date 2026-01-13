from models.point_cloud_registration.parenet.parenet_model import PARENetModel


model_cfg = {
    'class': PARENetModel,
    'args': {
        # Model architecture parameters
        'num_points_in_patch': 32,
        'ground_truth_matching_radius': 0.05,

        # Backbone parameters (PAREConv Feature Pyramid Network)
        'backbone_init_dim': 96,
        'backbone_output_dim': 256,
        'backbone_kernel_size': 4,
        'backbone_share_nonlinearity': False,
        'backbone_conv_way': 'edge_conv',
        'backbone_use_xyz': True,

        # Fine matching parameters
        'use_encoder_re_feats': True,

        # GeoTransformer parameters
        'geotransformer_input_dim': 768,
        'geotransformer_output_dim': 128,
        'geotransformer_hidden_dim': 96,
        'geotransformer_num_heads': 4,
        'geotransformer_blocks': ['self', 'cross', 'self', 'cross', 'self', 'cross'],
        'geotransformer_sigma_d': 4.8,
        'geotransformer_sigma_a': 15,
        'geotransformer_angle_k': 3,
        'geotransformer_reduction_a': 'max',

        # Coarse matching parameters
        'coarse_matching_num_targets': 128,
        'coarse_matching_overlap_threshold': 0.1,
        'coarse_matching_num_correspondences': 256,
        'coarse_matching_dual_normalization': True,

        # Fine matching parameters
        'fine_matching_topk': 2,
        'fine_matching_acceptance_radius': 0.6,
        'fine_matching_confidence_threshold': 0.005,
        'fine_matching_num_hypotheses': 1000,
        'fine_matching_num_refinement_steps': 5,
    },
}