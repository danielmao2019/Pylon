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
from configs.common.datasets.multi_task_learning.celeb_a import config as dataset_config
config.update(dataset_config)

# model config
from configs.common.models.multi_task_learning.celeb_a.celeb_a_resnet18 import model_config_all_tasks as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.multi_task_learning.graddrop import optimizer_config
from configs.common.optimizers.standard import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 91557264
config['train_seeds'] = [92188058, 92631038, 10482716, 20478406, 69582948, 48326882, 89931532, 48564734, 32402351, 39348277, 24086579, 95078172, 20612761, 92202988, 47679560, 91129663, 92090141, 26821207, 10746671, 33300681, 25118204, 63187797, 979682, 5200147, 95152621, 7582563, 14130139, 6906719, 16642649, 27285415, 41731880, 73275195, 63952628, 38062815, 6995234, 48360916, 26126031, 30212595, 97329369, 80784711, 21795822, 62182361, 56365563, 90988486, 98375655, 59489635, 5948885, 63567316, 61547169, 72238286, 1003935, 38198269, 40673749, 62522551, 56064194, 683497, 41546449, 92696637, 78579758, 9815723, 77887193, 67393622, 34122107, 70401810, 4695549, 46665698, 47220546, 20023896, 46352427, 64249368, 26635739, 59915157, 80923801, 88189701, 41511273, 74627570, 5975417, 58671590, 35992700, 62716137, 90316187, 70069028, 460271, 41575581, 86357864, 64659811, 3055227, 11700067, 98589827, 53604811, 69600482, 81360531, 41449121, 79195572, 9343030, 30978420, 53066171, 83753234, 32567961, 71889547]

# work dir
config['work_dir'] = "./logs/benchmarks/multi_task_learning/celeb_a/celeb_a_resnet18/graddrop_run_2"
