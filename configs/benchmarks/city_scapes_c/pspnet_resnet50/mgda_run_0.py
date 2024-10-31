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
config['init_seed'] = 44304457
config['train_seeds'] = [66833521, 12334661, 36534139, 42631479, 89709217, 57862606, 10976461, 92058722, 50709484, 36281677, 11466549, 11181325, 88826087, 61880084, 52411647, 26971776, 67717320, 2652775, 62660067, 39445450, 27512815, 12731822, 80249299, 14443371, 23016894, 91552245, 80882397, 66696021, 91438177, 85848454, 1501448, 2023603, 47556766, 5859779, 49064461, 73123929, 6644385, 81627018, 88256318, 43809735, 47246383, 16434880, 84682897, 19918401, 93423936, 80454477, 51431963, 31364347, 77470915, 50953950, 78698291, 78336613, 12588792, 90924530, 97670328, 52568187, 98924980, 62443276, 83069762, 82497111, 63923878, 85618818, 49882000, 91275694, 86262111, 92900341, 28476921, 5857954, 29444699, 28294651, 92539905, 21169167, 68819414, 44290379, 67842780, 72921422, 53733595, 41474197, 75849243, 70240277, 78015351, 18610497, 74675588, 57590905, 52910629, 44015387, 54053479, 73205944, 82485347, 53517075, 26636823, 39028147, 58533869, 39716356, 13798341, 94911532, 14802852, 48107151, 19029260, 21935817]

# work dir
config['work_dir'] = "./logs/benchmarks/city_scapes_c/pspnet_resnet50/mgda_run_0"
