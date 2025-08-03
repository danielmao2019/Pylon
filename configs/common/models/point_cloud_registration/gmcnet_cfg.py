"""
GMCNet model configuration for point cloud registration.

This config exactly matches the original GMCNet gmcnet_mn40_psc.yaml configuration
for reproducible results on ModelNet40 partial point cloud registration.
"""

from models.point_cloud_registration.gmcnet.gmcnet_wrapper import GMCNet

# GMCNet arguments matching original gmcnet_mn40_psc.yaml exactly
gmcnet_args = {
    # Network architecture parameters (exact match to original)
    'knn_list': '5,5,5',  # K-nearest neighbors for each hierarchical level
    'down_sample_list': '2,4',  # Downsampling ratios: level1/2, level2/4  
    'feature_size_list': '128,128,32,8',  # Feature dimensions: [128,128,32,8]
    'descriptor_size': 128,  # Final descriptor dimensionality
    
    # Loss configuration (exact weights from original)
    'use_cycle_loss': False,  # Original: False
    'use_mse_loss': 0.0,      # Original: 0.0 (numerical weight)
    'use_cd_loss': 1.0,       # Original: 1.0 (main loss)
    'use_inliers_loss': 0.01, # Original: 0.01 (regularization)
    'reductioin': 'mean',     # Original typo preserved for compatibility
    'use_annealing': False,   # Original: False
    
    # Feature configuration (exact match to original)
    'rif_only': False,              # Original: False
    'rif_feature': 'ppf',           # Original: ppf (Point Pair Features)
    'predict_inliers': False,       # Original: False  
    'use_weighted_procrustes': False,  # Original: False
}

model_cfg = {
    'class': GMCNet,
    'args': {
        'args': gmcnet_args,
    },
}
