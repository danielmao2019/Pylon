import models


model_config = {
    'class': models.change_detection.siamese_kpconv.SiameseKPConv,
    'args': {
        'in_channels': 4,  # Input channels (3 for XYZ + 1 for ones feature)
        'out_channels': 7,  # Urb3DCD dataset has 7 classes
        'point_influence': 0.05,  # Kernel point influence distance
        'down_channels': [64, 128, 256, 512, 1024],  # Following original: in_feat -> 2*in_feat -> 4*in_feat -> 8*in_feat -> 16*in_feat
        'up_channels': [1024, 512, 256, 128, 64],  # Following original: 16*in_feat -> 8*in_feat -> 4*in_feat -> 2*in_feat -> in_feat
        'bn_momentum': 0.02,  # Batch normalization momentum
        'dropout': 0.5,  # Dropout rate from original
        'conv_type': 'simple',  # Using SimpleBlock as in original first layer
        'block_params': {
            'n_kernel_points': 25,  # From original config
            'max_num_neighbors': 25,  # From original config
        },
    },
}
