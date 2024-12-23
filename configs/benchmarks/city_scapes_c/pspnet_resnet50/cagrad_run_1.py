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
from configs.common.optimizers.cagrad import optimizer_config
from configs.common.optimizers._core_ import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 31746027
config['train_seeds'] = [79897647, 89061074, 1476026, 46386028, 91214483, 35211652, 46056523, 90408569, 99278919, 86674293, 43642841, 91478740, 5604853, 85304405, 58827202, 35898771, 14320423, 56966148, 58757101, 71359553, 47526506, 54908411, 98712609, 49348515, 19626629, 87270222, 5539867, 50798118, 9397022, 69483478, 25708001, 32139278, 26959258, 27067891, 75781923, 40404156, 1388235, 92210327, 57429155, 30753603, 51056390, 73371426, 76256452, 43945119, 25082274, 60032170, 19987431, 48143734, 23144680, 61853938, 45824867, 56272279, 48120723, 45770531, 77464976, 22731825, 93983282, 42222698, 43636737, 16087194, 3119872, 67833059, 23371741, 22958770, 73984022, 64325839, 9852670, 93451322, 68648330, 11653033, 85665331, 25839807, 39870873, 68487061, 9032390, 75559250, 31907377, 24520348, 55574016, 42033625, 99343234, 7033680, 81089363, 70538825, 66083527, 28130053, 19507786, 45597499, 17648681, 23622216, 71066138, 11584377, 89858623, 37319188, 10035694, 79305736, 7479377, 56728758, 60996155, 89209573]

# work dir
config['work_dir'] = "./logs/benchmarks/city_scapes_c/pspnet_resnet50/cagrad_run_1"
