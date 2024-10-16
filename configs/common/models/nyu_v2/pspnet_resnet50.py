import models


model_config_depth_estimation = {
    'class': models.NYUD_MT_PSPNet,
    'args': {
        'backbone': models.backbones.resnet50(weights='DEFAULT'),
        'in_channels': 2048,
        'tasks': set(['depth_estimation']),
        'return_shared_rep': False,
        'use_attention': False,
    },
}

model_config_normal_estimation = {
    'class': models.NYUD_MT_PSPNet,
    'args': {
        'backbone': models.backbones.resnet50(weights='DEFAULT'),
        'in_channels': 2048,
        'tasks': set(['normal_estimation']),
        'return_shared_rep': False,
        'use_attention': False,
    },
}

model_config_semantic_segmentation = {
    'class': models.NYUD_MT_PSPNet,
    'args': {
        'backbone': models.backbones.resnet50(weights='DEFAULT'),
        'in_channels': 2048,
        'tasks': set(['semantic_segmentation']),
        'return_shared_rep': False,
        'use_attention': False,
    },
}

model_config_all_tasks = {
    'class': models.NYUD_MT_PSPNet,
    'args': {
        'backbone': models.backbones.resnet50(weights='DEFAULT'),
        'in_channels': 2048,
        'tasks': set(["depth_estimation", "normal_estimation", "semantic_segmentation"]),
        'return_shared_rep': True,
        'use_attention': False,
    },
}
