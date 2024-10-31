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
from configs.common.datasets.city_scapes_c import config as dataset_config
config.update(dataset_config)

# model config
from configs.common.models.city_scapes_c.pspnet_resnet50 import model_config_all_tasks as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.imtl import optimizer_config
from configs.common.optimizers._core_ import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 74345626
config['train_seeds'] = [22910444, 31254235, 42477525, 41541976, 9581191, 81758700, 53816562, 26486068, 26522709, 64819213, 46281551, 9870676, 14908230, 53312909, 7234428, 37719183, 65544423, 36044825, 6884270, 64710528, 55030754, 79655825, 77725197, 45245040, 26002968, 59140774, 26912576, 39093379, 91626005, 73982046, 44002844, 88167639, 55216734, 52954605, 94452854, 79862171, 24776969, 36296809, 28781030, 50367712, 96527951, 69749971, 13923850, 33519151, 70123859, 39450585, 83404411, 14276313, 31672425, 74323836, 20752242, 28430491, 92832836, 51876338, 460606, 4866622, 44307346, 1500826, 42387894, 85370436, 44121321, 97172729, 95890129, 76370207, 24184123, 4125822, 94734692, 45432259, 90663313, 86284400, 49761023, 47310762, 29708427, 64276804, 55961193, 40136861, 53401144, 11758429, 75854889, 45611377, 3013765, 86735488, 42083079, 5728489, 79464360, 6870931, 21437685, 16512118, 70344856, 33947868, 62791765, 23192928, 59335433, 14380742, 92382550, 72952888, 61138917, 88452651, 82737995, 60740435]

# work dir
config['work_dir'] = "./logs/benchmarks/city_scapes_c/pspnet_resnet50/imtl_run_1"
