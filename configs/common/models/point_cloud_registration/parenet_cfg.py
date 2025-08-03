from models.point_cloud_registration.parenet.parenet_model import PARENetModel


model_cfg = {
    'class': PARENetModel,
    'args': {
        # Backbone parameters (PAREConv Feature Pyramid Network)
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
        'output_dim': 256,
        
        # Position-Aware Rotation-Equivariant parameters
        'pare_conv_channels': [32, 64, 128, 256],
        'vn_backbone_output_dim': 256,
        
        # Coarse matching parameters
        'coarse_num_targets': 128,
        'coarse_overlap_threshold': 0.1,
        'coarse_num_correspondences': 256,
        'coarse_dual_normalization': True,
        
        # Fine matching parameters
        'fine_topk': 3,
        'fine_acceptance_radius': 0.1,
        'fine_mutual': True,
        'fine_confidence_threshold': 0.05,
        'fine_use_dustbin': False,
        'fine_use_global_score': False,
        'fine_correspondence_threshold': 3,
        'fine_correspondence_limit': None,
        'fine_num_refinement_steps': 5,
        
        # Ground truth matching parameters
        'ground_truth_matching_radius': 0.05,
        'num_points_in_patch': 32,
        'num_sinkhorn_iterations': 100,
    },
}