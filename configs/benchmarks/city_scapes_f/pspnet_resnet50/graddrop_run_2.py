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
from configs.common.optimizers.graddrop import optimizer_config
from configs.common.optimizers._core_ import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 72295
config['train_seeds'] = [85676965, 79128074, 10408749, 68946449, 62411967, 82981390, 78130530, 75367744, 72499115, 95326476, 392318, 9340885, 25944844, 7170717, 42234929, 27356768, 3954178, 83078497, 76236123, 18944784, 36366794, 15127441, 4986656, 87176854, 90308073, 82411643, 52364514, 33567706, 29959774, 89878585, 50705210, 41188013, 82864191, 68421773, 79266088, 89746772, 44258128, 47545795, 31841337, 2722561, 86320780, 16249619, 66001491, 92061065, 56884909, 74141121, 11341192, 8229609, 54904092, 26621071, 57939015, 58471525, 7195164, 31167746, 16636868, 15508683, 24722506, 56012714, 27616455, 37970061, 49310934, 20010926, 39903392, 61951608, 87375382, 70043943, 20924326, 44810324, 32353410, 72829103, 65656929, 80376348, 45970282, 86444921, 99713335, 18340097, 33439157, 58404352, 90356187, 61295804, 55455480, 76617552, 27938581, 87510426, 87610636, 63375055, 43993514, 895167, 91347095, 203061, 84130168, 2651479, 70574973, 51834330, 96212781, 89444004, 49650579, 78934928, 17160, 41516319]

# work dir
config['work_dir'] = "./logs/benchmarks/city_scapes_f/pspnet_resnet50/graddrop_run_2"
