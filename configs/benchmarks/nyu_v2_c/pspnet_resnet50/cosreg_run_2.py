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
from configs.common.datasets.nyu_v2_c import config as dataset_config
config.update(dataset_config)

# model config
from configs.common.models.nyu_v2_c.pspnet_resnet50 import model_config_all_tasks as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.cosreg import optimizer_config
from configs.common.optimizers._core_ import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 89230558
config['train_seeds'] = [99316339, 79230709, 86298705, 65152215, 6487764, 9723320, 36453464, 90333256, 67654178, 48828275, 60021996, 53932188, 62117554, 50717330, 76663161, 3899152, 90131367, 57714707, 13841953, 5135708, 59696892, 61360021, 33139853, 2036539, 156723, 83046257, 36114184, 95756490, 77112605, 88045513, 11038773, 22099810, 31022321, 14379033, 76808175, 27020312, 65877505, 28252648, 25514816, 45386726, 52935640, 24993481, 62895740, 14454629, 6934565, 9700047, 44584721, 49768428, 13617297, 25618636, 27226102, 84055986, 23988975, 38768834, 92603053, 98050414, 22354813, 21729939, 52480451, 90373329, 63682882, 69021402, 13396710, 87315635, 68439537, 31335411, 20211013, 46270672, 68117911, 38643001, 73209940, 84507107, 55854303, 21997089, 75354629, 31251942, 42803126, 47473697, 28891570, 13739385, 13154150, 71600125, 90356654, 3485849, 77212929, 99244350, 49844372, 42563393, 19072845, 48642656, 90585275, 79391481, 21942408, 2687444, 34041745, 34156618, 72101696, 43453296, 55993879, 96405312]

# work dir
config['work_dir'] = "./logs/benchmarks/nyu_v2_c/pspnet_resnet50/cosreg_run_2"
