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
from configs.common.datasets.nyu_v2_f import config as dataset_config
config.update(dataset_config)

# model config
from configs.common.models.nyu_v2_f.pspnet_resnet50 import model_config_all_tasks as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.rgw import optimizer_config
from configs.common.optimizers._core_ import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 79703350
config['train_seeds'] = [88533885, 82669911, 51479182, 838769, 23417610, 91453648, 57181842, 67894880, 79801837, 5106415, 29738200, 3731155, 48005575, 41003343, 53597432, 53362094, 15463849, 68327128, 56557186, 98767558, 53598191, 93797448, 22331998, 86103968, 80364146, 8296643, 30199040, 1392375, 59726034, 96372340, 49861946, 12262641, 29063214, 98575222, 47540275, 81050258, 57647783, 35449455, 60176400, 871340, 32577372, 74087727, 18552669, 48802322, 24170365, 5304696, 82703187, 19098964, 34890524, 68907359, 58074449, 90373874, 43872156, 15354809, 64241928, 75056373, 32660425, 48809553, 78095335, 37857410, 61051784, 48242434, 43541112, 40457829, 16151469, 82923866, 15033433, 73304625, 73466415, 31166885, 82881593, 97426920, 97755943, 84390262, 39200716, 91336372, 49543106, 54091844, 52548828, 29655712, 60232073, 46463626, 27472787, 45957783, 33558911, 10081651, 69603425, 62142349, 27269274, 89266404, 1046735, 52214501, 15006715, 82280726, 12193261, 4962902, 56926544, 16904063, 37890183, 11202640]

# work dir
config['work_dir'] = "./logs/benchmarks/nyu_v2_f/pspnet_resnet50/rgw_run_1"
