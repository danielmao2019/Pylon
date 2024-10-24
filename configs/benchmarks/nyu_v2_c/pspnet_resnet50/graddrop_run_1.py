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
from configs.common.optimizers.graddrop import optimizer_config
from configs.common.optimizers._core_ import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 21558386
config['train_seeds'] = [21002982, 15728335, 51673527, 61617811, 81209511, 29185727, 86498538, 2282638, 10461955, 6845789, 55953090, 31814019, 48880699, 59540064, 80617467, 64166546, 92400169, 31750364, 72443259, 14815873, 85292694, 89738565, 38215477, 31736781, 47787867, 90726941, 96236076, 98797859, 41735443, 20952699, 73054476, 54512204, 36987586, 54643128, 64442772, 24633652, 2390762, 75145040, 89279287, 52531069, 28657454, 90044987, 46612153, 13078089, 90776654, 3379914, 44908375, 86871110, 49313639, 34908722, 54130263, 46207569, 46071389, 58183721, 68234483, 91996753, 49761005, 69077324, 32128689, 85216457, 1334593, 56084835, 95378329, 31589119, 85643219, 93521313, 91386793, 62303820, 1590256, 77101849, 52504614, 75966936, 36443916, 48304285, 12909022, 38037119, 47188803, 6172609, 12117495, 8703904, 74445958, 35939672, 14707485, 74237634, 46896007, 6704591, 10013398, 72719960, 51121854, 24136606, 67946462, 40024363, 25432412, 70264234, 84189272, 91416613, 83986695, 57616370, 44106532, 48808515]

# work dir
config['work_dir'] = "./logs/benchmarks/nyu_v2_c/pspnet_resnet50/graddrop_run_1"
