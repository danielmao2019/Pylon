import models


mit_cfg = {
    'class': models.change_detection.changer.modules.ia_mix_vision_transformer.IA_MixVisionTransformer,
    'args': {
        'in_channels': 3,
        'embed_dims': 32,
        'num_stages': 4,
        'num_layers': [2, 2, 2, 2],
        'num_heads': [1, 2, 5, 8],
        'patch_sizes': [7, 3, 3, 3],
        'sr_ratios': [8, 4, 2, 1],
        'out_indices': (0, 1, 2, 3),
        'mlp_ratio': 4,
        'qkv_bias': True,
        'drop_rate': 0.0,
        'attn_drop_rate': 0.0,
        'drop_path_rate': 0.1,
    },
}

r18_cfg = {
    'class': models.change_detection.changer.modules.ia_resnet.IA_ResNetV1c,
    'args': {
        'depth': 18,
        'num_stages': 4,
        'out_indices': (0, 1, 2, 3),
        'dilations': (1, 1, 1, 1),
        'strides': (1, 2, 2, 2),
        'norm_cfg': dict(type='SyncBN', requires_grad=True),
        'norm_eval': False,
        'style': 'pytorch',
        'contract_dilation': True,
    },
}

s50_cfg = {
    'class': models.change_detection.changer.modules.ia_resnest.IA_ResNeSt,
    'args': {
        'depth': 50,
        'num_stages': 4,
        'out_indices': (0, 1, 2, 3),
        'dilations': (1, 1, 1, 1),
        'strides': (1, 2, 2, 2),
        'norm_cfg': dict(type='SyncBN', requires_grad=True),
        'norm_eval': False,
        'style': 'pytorch',
        'contract_dilation': True,
        'stem_channels': 64,
        'radix': 2,
        'reduction_factor': 4,
        'avg_down_stride': True,
    },
}

decoder_cfg = {
    'class': models.change_detection.changer.modules.changer_decoder.ChangerDecoder,
    'args': {
        'in_channels': [32, 64, 160, 256],
        'in_index': [0, 1, 2, 3],
        'channels': 128,
        'dropout_ratio': 0.1,
        'num_classes': 2,
        'norm_cfg': dict(type='SyncBN', requires_grad=True),
        'align_corners': False,
    },
}

changer_mit_b0_cfg = {
    'encoder_cfg': mit_cfg,
    'decoder_cfg': decoder_cfg,
}
