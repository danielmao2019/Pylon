import models
import torch


model_config = {
    'class': models.change_detection.BaseNet_LWGANet_L2,
    'args': {
        'preptrained_path': '/pub7/yuchen/Pylon/models/change_detection/lwganet/lwganet_l2_e296.pth',
    },
    'class': torch.distributed.init_process_group,
    'args': {
        'backend' = 'nccl',
        'init_method' = 'tcp://127.0.0.1:23456',
        'rank' = '0',
        'world_size' = '1',
    },
}