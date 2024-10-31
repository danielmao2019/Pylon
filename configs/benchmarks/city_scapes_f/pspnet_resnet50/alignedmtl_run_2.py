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
from configs.common.optimizers.alignedmtl import optimizer_config
from configs.common.optimizers._core_ import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 70919474
config['train_seeds'] = [90012813, 22052158, 43495522, 7433773, 60318763, 45868104, 34097723, 15211660, 375501, 74489982, 96142923, 33714492, 47401919, 70130989, 28637709, 6986700, 90473052, 74950364, 97340675, 67833383, 98811685, 82652145, 64776130, 59485736, 58377553, 80050395, 75902263, 38975024, 97896274, 67698051, 74294741, 94989896, 67333822, 34159925, 23106164, 30483251, 34530220, 55260638, 31365874, 79724617, 87588781, 45968076, 29216417, 66221379, 47000693, 35721679, 60115381, 52846172, 51753980, 61275510, 23808861, 5394742, 97075092, 24313232, 14148314, 92108443, 88470253, 63253655, 70583277, 65199825, 92207294, 91204275, 85659477, 77117794, 68889255, 66167202, 99737250, 71847923, 2136038, 31418942, 13514126, 80825622, 6864199, 13398639, 42612738, 1709457, 73649407, 35843202, 39485078, 79542216, 58650510, 15958576, 94807772, 83683046, 78355045, 10303397, 42235077, 74316815, 94467252, 33209318, 24902641, 21541324, 80161621, 33164369, 29658739, 44369334, 10819650, 55520700, 41986394, 26208425]

# work dir
config['work_dir'] = "./logs/benchmarks/city_scapes_f/pspnet_resnet50/alignedmtl_run_2"
