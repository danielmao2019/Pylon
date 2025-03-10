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
from configs.common.datasets.multi_task_learning.nyu_v2_c import config as dataset_config
config.update(dataset_config)

# model config
from configs.common.models.multi_task_learning.nyu_v2_c.pspnet_resnet50 import model_config_all_tasks as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.multi_task_learning.mgda_ub import optimizer_config
from configs.common.optimizers.standard import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 30206140
config['train_seeds'] = [19558334, 59408268, 59283205, 56815788, 67098825, 12277629, 70964819, 85714824, 51095363, 60685762, 9325956, 51767074, 15846665, 79206638, 66927778, 82220171, 22149525, 92862910, 22787672, 80473822, 16385856, 2800672, 84054013, 13134374, 63186331, 94328516, 79005735, 32574577, 77961527, 10034758, 99556502, 4366413, 92708992, 48362938, 79583288, 51182355, 96330433, 89280062, 10285717, 62455750, 33000646, 30037489, 20882071, 67690734, 701091, 79022119, 513084, 75211127, 14089992, 70171392, 38617842, 79543886, 25934411, 61697950, 70403676, 16932633, 89969618, 81117462, 7160553, 39847809, 98315669, 64857619, 4745698, 92911693, 22540386, 86472135, 53947398, 39742022, 37474304, 78559854, 77240473, 90290522, 8500569, 82749008, 21935217, 68466661, 95221724, 28423789, 41289846, 65077630, 46851391, 81630422, 34004951, 66427439, 47540279, 37391587, 49992639, 77747513, 3829145, 51805810, 61433329, 87866171, 7070067, 93112525, 72379254, 41255267, 81829378, 29377622, 33230362, 23312878]

# work dir
config['work_dir'] = "./logs/benchmarks/multi_task_learning/nyu_v2_c/pspnet_resnet50/mgda_ub_run_0"
