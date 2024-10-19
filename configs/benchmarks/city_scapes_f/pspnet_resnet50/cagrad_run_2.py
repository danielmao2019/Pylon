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
from configs.common.optimizers.cagrad import optimizer_config
from configs.common.optimizers._core_ import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 48708334
config['train_seeds'] = [95158823, 94932848, 46506242, 80064230, 59782586, 9575850, 18357393, 44710507, 44154916, 24781524, 20875911, 45050856, 10430279, 93606571, 3840280, 15840972, 28051118, 74632353, 86524021, 66601653, 77036414, 32559235, 14222348, 14912174, 33829900, 88844079, 32290464, 50261450, 96395392, 20344810, 69265067, 78945229, 72329322, 58525804, 55358196, 14010921, 64272193, 21794606, 57745520, 46033766, 47973180, 18898651, 52722040, 68188588, 95078307, 56767853, 74457677, 10648975, 88450501, 32117940, 69684440, 11969795, 49293456, 24547802, 88765414, 27666749, 3873898, 44212593, 39186393, 41716003, 49423023, 99561058, 10532126, 71710504, 93614878, 74443166, 14926177, 96106908, 6464560, 79160056, 40310144, 32836217, 67561298, 50614787, 47860497, 8943347, 47200560, 74941958, 17946495, 52189765, 3498918, 49949856, 21985754, 21289670, 17149638, 66495686, 98390464, 90692912, 22340913, 57269793, 41346781, 30217875, 5394317, 48176228, 38841629, 52600651, 89664472, 9902915, 77189164, 72393387]

# work dir
config['work_dir'] = "./logs/benchmarks/city_scapes_f/pspnet_resnet50/cagrad_run_2"
