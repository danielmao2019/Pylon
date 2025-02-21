import models
import deepcopy


model_cfg_template = {
    'class': models.change_detection.STMambaBCD,
    'args': {
        # common
        'patch_size': 4,
        'in_chans': 3,
        'ssm_rank_ratio': 2.0,
        'ssm_act_layer': "silu",
        'ssm_drop_rate': 0.0,
        'ssm_init': "v0",
        'mlp_act_layer': "gelu",
        'mlp_drop_rate': 0.0,
        'patch_norm': True,
        'norm_layer': "ln",
        'gmlp': False,
        'num_classes': 1000,  # this is used to initialize VSSM.classifier, which is unused
        # architecture-specific
        'pretrained': None,
        'drop_path_rate': None,
        'dims': None,
        'depths': None,
        'ssm_d_state': None,
        'ssm_dt_rank': None,
        'ssm_ratio': None,
        'ssm_conv': None,
        'ssm_conv_bias': None,
        'forward_type': None,
        'mlp_ratio': None,
        'downsample_version': None,
        'patchembed_version': None,
    }
}


model_base_cfg = deepcopy.copy(model_cfg_template)
model_base_cfg['args'].update({
    'pretrained': "./models/change_detection/change_mamba/checkpoints/vssm_base_0229_ckpt_epoch_237.pth",
    'drop_path_rate': 0.6,
    'dims': 128,
    'depths': [ 2, 2, 15, 2 ],
    'ssm_d_state': 1,
    'ssm_dt_rank': "auto",
    'ssm_ratio': 2.0,
    'ssm_conv': 3,
    'ssm_conv_bias': False,
    'forward_type': "v3noz",
    'mlp_ratio': 4.0,
    'downsample_version': "v3",
    'patchembed_version': "v2",
})

model_mini_cfg = deepcopy.copy(model_cfg_template)
model_mini_cfg['args'].update({
    'pretrained': None,  # not provided by the authors
    'drop_path_rate': 0.2,
    'dims': 96,
    'depths': [ 2, 2, 4, 2 ],
    'ssm_d_state': 1,
    'ssm_dt_rank': "auto",
    'ssm_ratio': 2.0,
    'ssm_conv': 3,
    'ssm_conv_bias': False,
    'forward_type': "v2",
    'mlp_ratio': -1.0,
    'downsample_version': "v3",
    'patchembed_version': "v2",
})

model_small_cfg = deepcopy.copy(model_cfg_template)
model_small_cfg['args'].update({
    'pretrained': "./models/change_detection/change_mamba/checkpoints/vssm_small_0229_ckpt_epoch_222.pth",
    'drop_path_rate': 0.3,
    'dims': 96,
    'depths': [ 2, 2, 15, 2 ],
    'ssm_d_state': 1,
    'ssm_dt_rank': "auto",
    'ssm_ratio': 2.0,
    'ssm_conv': 3,
    'ssm_conv_bias': False,
    'forward_type': "v3noz",
    'mlp_ratio': 4.0,
    'downsample_version': "v3",
    'patchembed_version': "v2",
})

model_tiny_cfg = deepcopy.copy(model_cfg_template)
model_tiny_cfg['args'].update({
    'pretrained': "./models/change_detection/change_mamba/checkpoints/vssm_tiny_0230_ckpt_epoch_262.pth",
    'drop_path_rate': 0.2,
    'dims': 96,
    'depths': [ 2, 2, 4, 2 ],
    'ssm_d_state': 1,
    'ssm_dt_rank': "auto",
    'ssm_ratio': 2.0,
    'ssm_conv': 3,
    'ssm_conv_bias': False,
    'forward_type': "v3noz",
    'mlp_ratio': 4.0,
    'downsample_version': "v3",
    'patchembed_version': "v2",
})
