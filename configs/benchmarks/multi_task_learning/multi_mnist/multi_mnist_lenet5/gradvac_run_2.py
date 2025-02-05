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
from configs.common.datasets.multi_mnist import config as dataset_config
config.update(dataset_config)

# model config
from configs.common.models.multi_mnist.multi_mnist_lenet5 import model_config_all_tasks as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.gradvac import optimizer_config
from configs.common.optimizers.standard import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 57661655
config['train_seeds'] = [11134441, 36340144, 28326793, 33035323, 55472862, 83803559, 49275608, 71596983, 59914719, 14724706, 46837179, 63257752, 59738693, 93895858, 52458632, 90876500, 90583478, 10681743, 62267545, 73399120, 6407146, 9923503, 91424946, 47116020, 30327143, 56298505, 53049287, 11420530, 53144927, 5266250, 11956498, 29205441, 5685213, 90288136, 19906943, 1301765, 80696514, 47307441, 63676671, 68195069, 85349658, 81279713, 43507250, 38181638, 72200996, 19818012, 65980403, 73671793, 70768476, 32763721, 89248577, 98444979, 22316532, 94053234, 44831552, 6918023, 65149037, 96759544, 8659417, 11986197, 3640401, 22884159, 93134778, 45128436, 81575579, 74547382, 41301738, 90068343, 89706906, 20835073, 58736282, 81641364, 62116860, 6753693, 39217878, 62597468, 1318903, 90221692, 35016464, 86345842, 63353643, 11074474, 53521991, 3527768, 97291428, 87246202, 39169025, 45139119, 81073988, 19912579, 83701225, 31478511, 9601530, 39659978, 19909556, 36681021, 70232597, 36906244, 26281793, 24792]

# work dir
config['work_dir'] = "./logs/benchmarks/multi_mnist/multi_mnist_lenet5/gradvac_run_2"
