import models
import torch_points3d


bn_momentum = 0.02
FEAT = 0
in_feat = 64
in_grid_size = 1

model_config = {
    'class': models.change_detection.siamese_kpconv_variants.OneConvFusionKPConv,
    'args': {
        'option': {
            'conv_type': 'PARTIAL_DENSE',
            'down_conv': {
                'class': torch_points3d.modules.KPConv.KPDualBlock,
                'n_kernel_points': 25,
                'down_conv_nn': [
                    [[FEAT + 1, in_feat], [in_feat, 2*in_feat]],
                    [[2*in_feat, 2*in_feat], [2*in_feat, 4*in_feat]],
                    [[4*in_feat, 4*in_feat], [4*in_feat, 8*in_feat]],
                    [[8*in_feat, 8*in_feat], [8*in_feat, 16*in_feat]],
                    [[16*in_feat, 16*in_feat], [16*in_feat, 32*in_feat]],
                ],
                'grid_size': [
                    [in_grid_size, in_grid_size],
                    [2*in_grid_size, 2*in_grid_size],
                    [4*in_grid_size, 4*in_grid_size],
                    [8*in_grid_size, 8*in_grid_size],
                    [16*in_grid_size, 16*in_grid_size],
                ],
                'prev_grid_size': [
                    [in_grid_size, in_grid_size],
                    [in_grid_size, 2*in_grid_size],
                    [2*in_grid_size, 4*in_grid_size],
                    [4*in_grid_size, 8*in_grid_size],
                    [8*in_grid_size, 16*in_grid_size],
                ],
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
                'deformable': [
                    [False, False],
                    [False, False],
                    [False, False],
                    [False, False],
                    [False, False]
                ],
                'max_num_neighbors': [[25, 25], [25, 30], [30, 38], [38, 38], [38, 38]],
            },
            'up_conv': {
                'class': torch_points3d.core.base_conv.partial_dense.FPModule_PD,
                'up_conv_nn': [
                    [32*in_feat + 16*in_feat, 8*in_feat],
                    [8*in_feat + 8*in_feat, 4*in_feat],
                    [4*in_feat + 4*in_feat, 2*in_feat],
                    [2*in_feat + 2*in_feat, in_feat],
                ],
                'skip': True,
                'up_k': [1, 1, 1, 1],
                'bn_momentum': [bn_momentum, bn_momentum, bn_momentum, bn_momentum, bn_momentum],
            },
            'mlp_cls': {
                'nn': [in_feat, in_feat],
                'dropout': 0.5,
                'bn_momentum': bn_momentum,
            },
        },
        'model_type': 'dummy',
        'num_classes': 7,
        'modules': models.change_detection.siamese_kpconv_variants.oneconvfusionkpconv,
    },
}
