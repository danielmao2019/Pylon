import models


model_config = {
    'class': models.change_detection.oneconvfusion_kpconv.OneConvFusionKPConv,
    'args': {
        'in_channels': 1,  # FEAT + 1 from original config
        'out_channels': 7,  # Number of classes, typically 7 for Urb3DCD
        'point_influence': 0.05,  # Derived from grid size
        'down_channels': [64, 128, 256, 512, 1024],  # Following original: in_feat -> 2*in_feat -> 4*in_feat -> 8*in_feat -> 16*in_feat -> 32*in_feat
        'up_channels': [1024, 512, 256, 128],  # Following original up_conv_nn
        'bn_momentum': 0.02,  # From original config
        'dropout': 0.5,  # From original mlp_cls
        'conv_type': 'dual',  # Using KPDualBlock as in original
        'block_params': {
            'n_kernel_points': 25,
            'block_names': [
                ["SimpleBlock", "ResnetBBlock"],
                ["ResnetBBlock", "ResnetBBlock"],
                ["ResnetBBlock", "ResnetBBlock"],
                ["ResnetBBlock", "ResnetBBlock"],
                ["ResnetBBlock", "ResnetBBlock"],
            ],
            'has_bottleneck': [
                [False, True],
                [True, True],
                [True, True],
                [True, True],
                [True, True]
            ],
            'max_num_neighbors': [
                [25, 25],
                [25, 30],
                [30, 38],
                [38, 38],
                [38, 38]
            ],
            'deformable': [
                [False, False],
                [False, False],
                [False, False],
                [False, False],
                [False, False]
            ],
        },
        'up_block_params': {
            'up_k': [1, 1, 1, 1],
            'skip': True,
        },
    },
}
