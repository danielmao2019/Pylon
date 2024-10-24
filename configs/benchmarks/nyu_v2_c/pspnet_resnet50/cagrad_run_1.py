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
from configs.common.optimizers.cagrad import optimizer_config
from configs.common.optimizers._core_ import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 46185056
config['train_seeds'] = [669754, 9944186, 6289368, 99974263, 65010747, 88463620, 19048864, 97548379, 68972380, 7169606, 84895519, 73391287, 4889280, 56103182, 71109762, 22197861, 45203115, 72897885, 60519718, 21621438, 77882861, 10758109, 28759002, 84810945, 97856314, 84887536, 97669786, 64277441, 69437636, 26517318, 15666296, 42356264, 84488056, 69439880, 41332648, 2153682, 61651029, 32531784, 45662684, 12319008, 6836591, 28835173, 96124871, 33960039, 69316569, 49474954, 97294424, 22657487, 88596918, 16414524, 64730847, 6039204, 19134024, 27597373, 31617109, 32889742, 1013022, 90307309, 77015898, 38823595, 50444954, 22144785, 66536824, 37843153, 46453005, 97779827, 44958538, 25624345, 95366769, 99684369, 48858405, 27837507, 18401754, 7749937, 63688630, 97383092, 8136770, 92599372, 74353128, 59456108, 57993353, 8107993, 37764287, 35207179, 20936562, 66575516, 10536939, 87920650, 29244898, 93176086, 64636873, 33644889, 55929774, 15597397, 58854153, 29386273, 29017031, 82528210, 54471838, 92250664]

# work dir
config['work_dir'] = "./logs/benchmarks/nyu_v2_c/pspnet_resnet50/cagrad_run_1"
