# This file is automatically generated by `./configs/benchmarks/multi_task_learning/gen_multi_task_learning.py`.
# Please do not attempt to modify manually.
import torch
import schedulers


config = {
    'runner': None,
    'work_dir': None,
    'epochs': 100,
    # seeds
    'init_seed': None,
    'train_seeds': None,
    # dataset config
    'train_dataset': None,
    'train_dataloader': None,
    'val_dataset': None,
    'val_dataloader': None,
    'test_dataset': None,
    'test_dataloader': None,
    'criterion': None,
    'metric': None,
    # model config
    'model': None,
    # optimizer config
    'optimizer': None,
    'scheduler': {
        'class': torch.optim.lr_scheduler.LambdaLR,
        'args': {
            'lr_lambda': {
                'class': schedulers.WarmupLambda,
                'args': {},
            },
        },
    },
}

from runners import SupervisedMultiTaskTrainer
config['runner'] = SupervisedMultiTaskTrainer

# dataset config
from configs.common.datasets.multi_task_learning.nyu_v2_f import config as dataset_config
config.update(dataset_config)

# model config
from configs.common.models.multi_task_learning.nyu_v2_f.pspnet_resnet50 import model_config_all_tasks as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.multi_task_learning.graddrop import optimizer_config
from configs.common.optimizers.standard import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 77776974
config['train_seeds'] = [63035009, 23166536, 51576180, 72840179, 27281073, 57772391, 2535869, 39059435, 71563509, 26587783, 25065026, 72276042, 69521651, 84922372, 43207580, 67058996, 59532934, 30956158, 21509228, 62675320, 95368357, 26690327, 47156806, 21279241, 46069921, 79027446, 90462305, 61140703, 95903621, 82515840, 54453246, 89288968, 22228245, 91360636, 49393046, 96976020, 53941387, 23605599, 66866364, 47753280, 64253231, 76056945, 34917921, 56092584, 84286853, 54841293, 3269260, 13743134, 84586483, 90796416, 16058696, 8711425, 33341836, 54358659, 97791305, 91895381, 86326784, 97118970, 57517232, 58610786, 78753666, 30562307, 65895097, 49617434, 5104918, 28313814, 62620977, 19241089, 83214294, 50109316, 55114201, 15567194, 77701173, 49412034, 17591120, 54427308, 80018399, 99550165, 56841849, 79960399, 4698327, 48686053, 38196570, 65916063, 87790236, 63000418, 43498779, 33843844, 72910688, 43018183, 92468156, 33446500, 31109778, 42612780, 34282492, 15750749, 80111853, 49738764, 82487283, 60798712]

# work dir
config['work_dir'] = "./logs/benchmarks/multi_task_learning/nyu_v2_f/pspnet_resnet50/graddrop_run_1"
