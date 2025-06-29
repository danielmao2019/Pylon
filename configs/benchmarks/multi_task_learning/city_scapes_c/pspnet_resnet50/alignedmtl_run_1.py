# This file is automatically generated by `./configs/benchmarks/multi_task_learning/gen_multi_task_learning.py`.
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
    'val_seeds': None,
    'test_seed': None,
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
                'class': schedulers.lr_lambdas.WarmupLambda,
                'args': {},
            },
        },
    },
}

from runners import SupervisedMultiTaskTrainer
config['runner'] = SupervisedMultiTaskTrainer

# dataset config
from configs.common.datasets.multi_task_learning.city_scapes_c import config as dataset_config
config.update(dataset_config)

# model config
from configs.common.models.multi_task_learning.city_scapes_c.pspnet_resnet50 import model_config_all_tasks as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.multi_task_learning.alignedmtl import optimizer_config
from configs.common.optimizers.standard import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 13198971
config['train_seeds'] = [81531394, 39861694, 86384313, 19510447, 53104053, 92623184, 14198718, 81647514, 58069998, 71411895, 1253933, 21651408, 68422607, 18892748, 32122397, 49350763, 38913160, 66039259, 18666680, 73063363, 7711117, 41052379, 25727714, 58007696, 40148218, 19871627, 84630362, 32934738, 59386532, 84488508, 54435257, 78249441, 4394354, 15290467, 18190876, 23031508, 78833817, 33935069, 45476434, 8456492, 44158987, 59625104, 74615755, 89753793, 39128158, 24141784, 84841022, 98714753, 47696915, 33285543, 71648351, 11397531, 73158415, 47178083, 48416048, 45105953, 90651029, 16426902, 74468329, 84518891, 84086168, 87069280, 2450367, 85650454, 62615836, 54296914, 83015402, 50857026, 91884777, 76736825, 10478526, 81578606, 31222979, 85933864, 51107557, 83859943, 28243139, 74429006, 1909927, 55325131, 76607507, 39005130, 96422545, 1904495, 93548037, 54332327, 63768787, 90917694, 78358412, 92841164, 76792019, 94356599, 68224218, 68941960, 94415761, 5524675, 1553789, 11982994, 40284232, 87593889]
config['val_seeds'] = [77246605, 71607009, 72976439, 85714476, 75687969, 45274706, 53513757, 54075045, 57690004, 94259633, 39948053, 63538607, 2035877, 16001251, 39006249, 43233835, 79277350, 20852656, 23445086, 1684451, 72089360, 41764310, 43402822, 87648617, 42151789, 19088549, 80485061, 96296793, 14242360, 83349590, 40878381, 72583603, 13825951, 45322341, 23849837, 22999073, 76406503, 11313956, 26263963, 92521552, 87562650, 58870360, 45922853, 6037626, 95017023, 18059193, 60161054, 8364477, 52110437, 94158490, 9474426, 38931003, 98692108, 34345486, 67000561, 95313545, 68256742, 77284827, 87991959, 16587199, 92622035, 62521759, 6464426, 12750786, 61993997, 18185350, 32567755, 3993594, 77594165, 93883944, 1917110, 52973598, 315527, 22785011, 91608949, 63141886, 76400218, 59118336, 73393969, 74123039, 33931252, 84782818, 3027345, 96433943, 61520726, 14774478, 36443955, 73742051, 76414316, 30761971, 74123200, 74590930, 25205899, 382642, 19664711, 68948052, 65946045, 68550636, 22364526, 32230544]
config['test_seed'] = 74621701

# work dir
config['work_dir'] = "./logs/benchmarks/multi_task_learning/city_scapes_c/pspnet_resnet50/alignedmtl_run_1"
