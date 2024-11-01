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
from configs.common.optimizers.mgda import optimizer_config
from configs.common.optimizers._core_ import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 76343487
config['train_seeds'] = [95856163, 82742064, 25300257, 25222540, 785721, 24401275, 26474803, 53218702, 7905261, 48320136, 93046842, 13306345, 59490469, 63481786, 90465068, 78698222, 42619061, 14536932, 29875183, 6640469, 30560622, 69669256, 29449020, 96320592, 22916531, 99327273, 292625, 6620218, 6462907, 6458614, 43564779, 77366583, 5755980, 17802738, 25381938, 10404049, 68556353, 64384296, 17586456, 39800050, 25558645, 34621313, 24403117, 8296601, 81204463, 9409802, 42284423, 35182209, 34669614, 20428773, 21574100, 54231440, 16097057, 30682646, 5732438, 10875135, 2653866, 19507889, 11407705, 7563942, 69978981, 30703361, 45783624, 17747514, 50062451, 38922277, 78510820, 87839739, 72458003, 85531353, 98402731, 46633806, 82824619, 28243143, 77615410, 75591783, 10173299, 88745830, 14098607, 6070154, 40883169, 1752237, 21986886, 84208079, 48058901, 31035918, 18615967, 47897725, 71821785, 66567343, 20096430, 63259304, 28902895, 64850194, 29595212, 47447876, 70162450, 72669959, 89596201, 7241814]

# work dir
config['work_dir'] = "./logs/benchmarks/celeb_a/celeb_a_resnet18/mgda_run_1"