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
from configs.common.optimizers.rgw import optimizer_config
from configs.common.optimizers._core_ import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 51781589
config['train_seeds'] = [90886106, 71258641, 20925038, 75676009, 71688865, 80446168, 88666557, 75320551, 44105788, 10473995, 59086749, 93408563, 96738012, 2709090, 11348521, 81775765, 87471781, 89625391, 90924644, 17205297, 70123280, 87683606, 20027990, 5365652, 50660949, 5749406, 33242575, 60442998, 821717, 55192589, 64394062, 85463396, 95915648, 51880668, 56198902, 82812451, 67750710, 88198032, 12593989, 63351327, 36371547, 82216363, 91286718, 31819206, 96414925, 2256239, 9955387, 77428857, 17551548, 48549962, 35554687, 16929681, 18881786, 7263364, 17428375, 32667265, 91894344, 24175941, 40324918, 15640027, 57472891, 41860548, 33378111, 65251359, 15521297, 29201071, 79806516, 77577779, 73005958, 69297744, 91444647, 95980193, 72762037, 68707040, 91686326, 34741667, 4716074, 36870134, 62071063, 61400953, 86638187, 14242870, 47089555, 1475321, 30356838, 52864133, 18544751, 95729918, 35919549, 23081985, 59438438, 30826882, 81202993, 15763203, 26417404, 41594693, 31911343, 32297623, 98633562, 8946997]

# work dir
config['work_dir'] = "./logs/benchmarks/city_scapes_f/pspnet_resnet50/rgw_run_1"
