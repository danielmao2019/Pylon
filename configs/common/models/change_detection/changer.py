import copy
import models
import mmseg


mit_cfg = {
    'class': models.change_detection.changer.modules.interaction_mit.IA_MixVisionTransformer,
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
    'class': models.change_detection.changer.modules.interaction_resnet.IA_ResNetV1c,
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
    'class': models.change_detection.changer.modules.interaction_resnest.IA_ResNeSt,
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

interaction_cfg = (
    None,
    dict(type='SpatialExchange', p=1/2),
    dict(type='ChannelExchange', p=1/2),
    dict(type='ChannelExchange', p=1/2),
)

sampler_cfg = dict(type='mmseg.OHEMPixelSampler', thresh=0.7, min_kept=100000)

# ==================================================
# mit b0
# ==================================================

changer_mit_b0_cfg = {
    'class': models.change_detection.changer.Changer,
    'args': {
        'encoder_cfg': copy.deepcopy(mit_cfg),
        'decoder_cfg': copy.deepcopy(decoder_cfg),
    },
}
changer_mit_b0_cfg['args']['encoder_cfg']['args'].update({
    'pretrained': "./models/change_detection/changer/checkpoints/mit_b0.pth",
    'interaction_cfg': interaction_cfg,
})
changer_mit_b0_cfg['args']['decoder_cfg']['args']['sampler'] = sampler_cfg

# ==================================================
# mit b1
# ==================================================

changer_mit_b1_cfg = copy.deepcopy(changer_mit_b0_cfg)
changer_mit_b1_cfg['args']['encoder_cfg']['args'].update({
    'pretrained': "./models/change_detection/changer/checkpoints/mit_b1.pth",
    'embed_dims': 64, 'num_heads': [1, 2, 5, 8], 'num_layers': [2, 2, 2, 2],
})
changer_mit_b1_cfg['args']['decoder_cfg']['args']['in_channels'] = [64, 128, 320, 512]

# ==================================================
# r18
# ==================================================

changer_r18_cfg = {
    'class': models.change_detection.changer.Changer,
    'args': {
        'encoder_cfg': copy.deepcopy(r18_cfg),
        'decoder_cfg': copy.deepcopy(decoder_cfg),
    },
}
changer_r18_cfg['args']['encoder_cfg']['args']['interaction_cfg'] = interaction_cfg
changer_r18_cfg['args']['decoder_cfg']['args']['sampler'] = sampler_cfg

# ==================================================
# s50
# ==================================================

changer_s50_cfg = {
    'class': models.change_detection.changer.Changer,
    'args': {
        'encoder_cfg': copy.deepcopy(s50_cfg),
        'decoder_cfg': copy.deepcopy(decoder_cfg),
    },
}
changer_s50_cfg['args']['encoder_cfg']['args']['interaction_cfg'] = interaction_cfg
changer_s50_cfg['args']['decoder_cfg']['args']['sampler'] = sampler_cfg

# ==================================================
# s101
# ==================================================

changer_s101_cfg = copy.deepcopy(changer_s50_cfg)
changer_s101_cfg['args']['encoder_cfg']['args'].update({'depth': 101, 'stem_channels': 128})
