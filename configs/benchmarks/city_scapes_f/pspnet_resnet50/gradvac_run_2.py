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
from configs.common.datasets.city_scapes_f import config as dataset_config
config.update(dataset_config)

# model config
from configs.common.models.city_scapes_f.pspnet_resnet50 import model_config_all_tasks as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.gradvac import optimizer_config
from configs.common.optimizers._core_ import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 24698937
config['train_seeds'] = [2810328, 37722599, 33756552, 95138273, 11021176, 78452151, 85385401, 4727449, 19185063, 6499300, 28222075, 924326, 57887962, 79160875, 32974578, 78499927, 62245350, 50253331, 80766715, 34559631, 60317269, 91124624, 80769771, 49122191, 89232171, 29204972, 82208729, 70626289, 40221672, 66601981, 94343716, 58546044, 58358673, 35548467, 55145759, 48200049, 63692979, 43556658, 17903782, 211828, 21960729, 46597049, 88403895, 56434168, 14404984, 25992273, 29281799, 65685972, 78056166, 61658030, 34490154, 16896386, 78330868, 70102451, 46875464, 71459564, 42733368, 82807016, 68476481, 10286504, 12245780, 95510708, 42165936, 41341086, 27503786, 73743103, 22746332, 33771347, 94020770, 92346226, 81983639, 23301699, 48330810, 57171882, 65283136, 59782068, 70485837, 4664021, 98766656, 8557614, 14068854, 88715785, 69134385, 46049309, 46618121, 7122437, 98604518, 83530189, 29718793, 85052501, 14566990, 59601697, 21437467, 33517549, 25142734, 26929807, 32208489, 9449858, 30915374, 41313647]

# work dir
config['work_dir'] = "./logs/benchmarks/city_scapes_f/pspnet_resnet50/gradvac_run_2"
