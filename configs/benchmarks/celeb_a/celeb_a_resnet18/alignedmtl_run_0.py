# This file is automatically generated by `./configs/benchmarks/multi_task_learning.py`.
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
from configs.common.datasets.celeb_a import config as dataset_config
config.update(dataset_config)

# model config
from configs.common.models.celeb_a.celeb_a_resnet18 import model_config_all_tasks as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.alignedmtl import optimizer_config
from configs.common.optimizers._core_ import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 97731171
config['train_seeds'] = [21892271, 77588852, 86409576, 93242687, 43538595, 42646575, 90982766, 32087170, 86337677, 21871652, 49383429, 74519921, 94586620, 53192335, 92972144, 1706909, 83038382, 46442717, 28367881, 67969786, 84325506, 76729043, 5178386, 5158855, 87473752, 87968199, 77992379, 95207433, 57943229, 35873619, 88023959, 96738716, 7090324, 87458995, 61478261, 21971824, 48901881, 81489208, 34013819, 90410268, 41518930, 97751437, 47666124, 66572330, 242411, 63154871, 60022687, 9668560, 43444506, 43243736, 13459689, 6193639, 96097712, 1709511, 34719116, 36904968, 53043623, 30838482, 44783234, 51275767, 93373592, 88707724, 3839626, 36295055, 79872187, 70147973, 96366279, 20588883, 48495103, 33979360, 75458532, 60906717, 85237883, 25548929, 99563021, 48549726, 67231643, 43147232, 24884269, 8513920, 86336637, 49594670, 4182406, 24545956, 51588321, 5509579, 36516434, 16033075, 10587775, 27876745, 23105575, 65203976, 85233133, 47326132, 49635927, 4594350, 59051304, 15864922, 59473082, 81417238]

# work dir
config['work_dir'] = "./logs/benchmarks/celeb_a/celeb_a_resnet18/alignedmtl_run_0"
