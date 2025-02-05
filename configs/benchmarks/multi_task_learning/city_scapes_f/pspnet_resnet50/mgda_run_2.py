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
from configs.common.optimizers.mgda import optimizer_config
from configs.common.optimizers.standard import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 33044742
config['train_seeds'] = [74672177, 82174156, 14502245, 2586317, 77791105, 13343459, 31635628, 15275466, 67396171, 58545016, 54249584, 46497663, 36849952, 15915067, 9041218, 56454417, 38875427, 11914973, 91729802, 19906930, 75961071, 98201500, 69288460, 7368839, 27152438, 20735183, 31978590, 79189690, 62727667, 79219222, 11922461, 57126419, 48513980, 64780648, 69172704, 9901186, 11056492, 40841414, 12149435, 56941279, 79478824, 54483054, 21712168, 7891561, 8980997, 87643227, 81323331, 72093912, 98428155, 19521257, 14092671, 31095630, 54939806, 16284334, 88751453, 75225834, 97912852, 11014274, 56493930, 78431193, 35553959, 12063487, 41893312, 44946920, 13685611, 74442515, 32038985, 61662190, 867634, 68539868, 4053340, 13106392, 9883891, 87306228, 95398698, 51106407, 33467642, 83172799, 72986414, 33150938, 20105444, 94909189, 94524581, 94845117, 29586434, 32062964, 84731323, 69478133, 76553422, 76243146, 43364214, 15645602, 72329023, 43598042, 46696112, 33002920, 7992411, 29413907, 1849543, 86870149]

# work dir
config['work_dir'] = "./logs/benchmarks/city_scapes_f/pspnet_resnet50/mgda_run_2"
