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
from configs.common.optimizers.mgda import optimizer_config
from configs.common.optimizers._core_ import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 13168113
config['train_seeds'] = [95037246, 27438393, 96656219, 87619650, 58018826, 11681606, 46401993, 10326407, 74092181, 98436436, 13512383, 93236175, 57756471, 43255746, 77929773, 82352837, 74807039, 36271796, 8396384, 73125614, 87253072, 5610012, 33880101, 28710324, 68693278, 18479764, 95705476, 47925492, 4239085, 46213911, 55951406, 79652133, 40676087, 10174350, 89284853, 58288147, 24604296, 31892107, 27146831, 47383010, 45423617, 66351888, 80501221, 94594598, 93673250, 58524564, 82665763, 97304670, 59526975, 18300700, 54770287, 77860558, 42301750, 92044285, 96981793, 83715220, 43276419, 38706893, 88342884, 71986671, 59376416, 60977336, 58750440, 23622770, 4605252, 50569023, 60342714, 24640461, 6362524, 20004248, 8045485, 21334971, 38188097, 86514474, 55592622, 41146955, 80117671, 92177584, 36878405, 25777268, 96976077, 51694854, 25514665, 14827878, 35746660, 20070773, 27774670, 11674086, 19089909, 46664621, 14298123, 19867689, 51469480, 19628131, 85504582, 78174165, 89174883, 25009769, 42814290, 3436026]

# work dir
config['work_dir'] = "./logs/benchmarks/city_scapes_c/pspnet_resnet50/mgda_run_2"
