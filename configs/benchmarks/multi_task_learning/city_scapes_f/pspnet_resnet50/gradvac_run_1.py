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
from configs.common.datasets.multi_task_learning.city_scapes_f import config as dataset_config
config.update(dataset_config)

# model config
from configs.common.models.multi_task_learning.city_scapes_f.pspnet_resnet50 import model_config_all_tasks as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.multi_task_learning.gradvac import optimizer_config
from configs.common.optimizers.standard import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 81557253
config['train_seeds'] = [13388406, 35132733, 9266380, 60260168, 35325767, 88693320, 82736209, 37240983, 82187629, 38567598, 92729731, 31303927, 85736164, 81698733, 69110140, 52312652, 68832496, 6746911, 16642811, 73511039, 62142680, 55034646, 14336473, 27781041, 48581425, 60468556, 41690010, 60760630, 61699671, 57466348, 39561983, 35580752, 37777319, 27353904, 69709425, 52721801, 39392312, 16578441, 28628979, 73833948, 98095466, 95744540, 51491150, 59893731, 91405274, 29217883, 37516013, 99511729, 82264091, 68938344, 73251931, 2599036, 65641222, 39104861, 17216085, 10369186, 45461887, 18368611, 83681201, 4998896, 62169405, 54957680, 60419413, 33651040, 66371635, 99950263, 14904819, 14529417, 15659194, 28208155, 60399687, 71795379, 40622142, 67921513, 59466240, 15459064, 43265111, 58479075, 83861392, 7293543, 55219816, 95973356, 77945319, 63518196, 64224217, 83021973, 97866676, 11845642, 52228046, 7038633, 78458937, 75406756, 37087214, 33136492, 32890523, 7180120, 7205679, 86131022, 99465163, 5454401]

# work dir
config['work_dir'] = "./logs/benchmarks/multi_task_learning/city_scapes_f/pspnet_resnet50/gradvac_run_1"
