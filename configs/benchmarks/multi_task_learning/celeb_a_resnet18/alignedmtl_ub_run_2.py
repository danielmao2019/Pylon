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
from configs.common.optimizers.alignedmtl_ub import optimizer_config
from configs.common.optimizers.standard import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 59924737
config['train_seeds'] = [40767774, 61779769, 6281187, 31137145, 57929205, 86541069, 13934843, 91751353, 14121197, 89532634, 12728106, 33463686, 69389862, 15494665, 21212216, 50893121, 69025040, 96953940, 40182793, 32021201, 18651489, 16909129, 48214123, 51643018, 15075887, 67758876, 78676065, 54597733, 41073995, 2462840, 28589637, 97263794, 41833074, 24551972, 65702081, 59367578, 30871944, 56033729, 7259005, 30721653, 18103353, 88584735, 26818141, 1609900, 28432461, 34934962, 28300333, 98690608, 47760542, 56529757, 81445042, 54042777, 90060254, 33063348, 31628662, 58594222, 50500375, 38289058, 86727390, 50826334, 11498571, 45550795, 17635240, 95075434, 6735980, 70602301, 51796060, 15835014, 58842314, 93498693, 1325751, 29762064, 7081097, 13945300, 94498827, 38060352, 72251452, 35799442, 8533404, 62853393, 60595152, 91399308, 29133309, 60299387, 19676531, 21728898, 37855623, 60874427, 94260018, 46289098, 33225095, 69937624, 29730097, 66749650, 89777813, 4331018, 53819942, 24749766, 27842095, 97080091]

# work dir
config['work_dir'] = "./logs/benchmarks/celeb_a/celeb_a_resnet18/alignedmtl_ub_run_2"
