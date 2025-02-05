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
from configs.common.optimizers.rgw import optimizer_config
from configs.common.optimizers.standard import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 42622742
config['train_seeds'] = [57290134, 74298539, 90052317, 59001706, 1051565, 94144192, 59992690, 15082793, 80170980, 24195345, 83412870, 64725618, 21803426, 69042758, 41745133, 9897045, 85613684, 21454154, 10579558, 84760166, 59181082, 77159272, 25345879, 22625240, 76181025, 17348143, 26516093, 5197316, 31487604, 62884799, 25312506, 71788227, 31051485, 19440313, 79431452, 26537442, 8117363, 85849454, 53323723, 93504839, 53582968, 83658373, 72100727, 31701907, 60487464, 94405121, 16564517, 98935134, 52313480, 95540407, 27260892, 60586478, 4230449, 4977399, 82419063, 86015507, 26701547, 50580228, 34937406, 61348370, 63140412, 44328172, 77559766, 60917219, 25809198, 20245580, 3401498, 62449703, 11695941, 70247087, 28810332, 10132926, 5594749, 23249284, 68972390, 33878222, 91994994, 51873244, 81906825, 14223664, 56716504, 7464607, 83902070, 80999427, 62639696, 11214460, 75544043, 82304412, 51399911, 16127002, 74644942, 79554908, 18319358, 69831430, 51160355, 75753661, 4593323, 24070537, 95803947, 37491277]

# work dir
config['work_dir'] = "./logs/benchmarks/celeb_a/celeb_a_resnet18/rgw_run_2"
