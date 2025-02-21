import models
import deepcopy


model_cfg_template = {
    'class': models.change_detection.STMambaBCD,
    'args': {
        # common
        'PATCH_SIZE': 4,
        'IN_CHANS': 3,
        'SSM_RANK_RATIO': 2.0,
        'SSM_ACT_LAYER': "silu",
        'SSM_DROP_RATE': 0.0,
        'SSM_INIT': "v0",
        'MLP_ACT_LAYER': "gelu",
        'MLP_DROP_RATE': 0.0,
        'PATCH_NORM': True,
        'NORM_LAYER': "ln",
        'GMLP': False,
        'num_classes': 1000,  # this is used to initialize VSSM.classifier, which is unused
        # architecture-specific
        'pretrained': None,
        'DROP_PATH_RATE': None,
        'EMBED_DIM': None,
        'DEPTHS': None,
        'SSM_D_STATE': None,
        'SSM_DT_RANK': None,
        'SSM_RATIO': None,
        'SSM_CONV': None,
        'SSM_CONV_BIAS': None,
        'SSM_FORWARDTYPE': None,
        'MLP_RATIO': None,
        'DOWNSAMPLE': None,
        'PATCHEMBED': None,
    }
}


model_base_cfg = deepcopy.copy(model_cfg_template)
model_base_cfg['args'].update({
    'pretrained': "./models/change_detection/change_mamba/checkpoints/vssm_base_0229_ckpt_epoch_237.pth",
    'DROP_PATH_RATE': 0.6,
    'EMBED_DIM': 128,
    'DEPTHS': [ 2, 2, 15, 2 ],
    'SSM_D_STATE': 1,
    'SSM_DT_RANK': "auto",
    'SSM_RATIO': 2.0,
    'SSM_CONV': 3,
    'SSM_CONV_BIAS': False,
    'SSM_FORWARDTYPE': "v3noz",
    'MLP_RATIO': 4.0,
    'DOWNSAMPLE': "v3",
    'PATCHEMBED': "v2",
})

model_mini_cfg = deepcopy.copy(model_cfg_template)
model_mini_cfg['args'].update({
    'pretrained': None,  # not provided by the authors
    'DROP_PATH_RATE': 0.2,
    'EMBED_DIM': 96,
    'DEPTHS': [ 2, 2, 4, 2 ],
    'SSM_D_STATE': 1,
    'SSM_DT_RANK': "auto",
    'SSM_RATIO': 2.0,
    'SSM_CONV': 3,
    'SSM_CONV_BIAS': False,
    'SSM_FORWARDTYPE': "v2",
    'MLP_RATIO': -1.0,
    'DOWNSAMPLE': "v3",
    'PATCHEMBED': "v2",
})

model_small_cfg = deepcopy.copy(model_cfg_template)
model_small_cfg['args'].update({
    'pretrained': "./models/change_detection/change_mamba/checkpoints/vssm_small_0229_ckpt_epoch_222.pth",
    'DROP_PATH_RATE': 0.3,
    'EMBED_DIM': 96,
    'DEPTHS': [ 2, 2, 15, 2 ],
    'SSM_D_STATE': 1,
    'SSM_DT_RANK': "auto",
    'SSM_RATIO': 2.0,
    'SSM_CONV': 3,
    'SSM_CONV_BIAS': False,
    'SSM_FORWARDTYPE': "v3noz",
    'MLP_RATIO': 4.0,
    'DOWNSAMPLE': "v3",
    'PATCHEMBED': "v2",
})

model_tiny_cfg = deepcopy.copy(model_cfg_template)
model_tiny_cfg['args'].update({
    'pretrained': "./models/change_detection/change_mamba/checkpoints/vssm_tiny_0230_ckpt_epoch_262.pth",
    'DROP_PATH_RATE': 0.2,
    'EMBED_DIM': 96,
    'DEPTHS': [ 2, 2, 4, 2 ],
    'SSM_D_STATE': 1,
    'SSM_DT_RANK': "auto",
    'SSM_RATIO': 2.0,
    'SSM_CONV': 3,
    'SSM_CONV_BIAS': False,
    'SSM_FORWARDTYPE': "v3noz",
    'MLP_RATIO': 4.0,
    'DOWNSAMPLE': "v3",
    'PATCHEMBED': "v2",
})
