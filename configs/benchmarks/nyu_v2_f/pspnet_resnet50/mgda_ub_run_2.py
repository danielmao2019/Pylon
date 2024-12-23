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
from configs.common.optimizers.mgda_ub import optimizer_config
from configs.common.optimizers._core_ import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 27616836
config['train_seeds'] = [1895251, 34628718, 16437047, 17186772, 29660667, 79993379, 66709034, 12873426, 96022024, 31766059, 42054395, 6715528, 80529715, 87011924, 89656070, 9204851, 77665820, 83164263, 77667937, 71582797, 44586648, 9196114, 18880766, 50109984, 38305026, 84313527, 32429637, 77749516, 54747927, 82163367, 16146458, 28774552, 88597457, 38242895, 93584273, 61622494, 71570906, 15197105, 85693522, 23154388, 25291699, 78017246, 59007968, 14052847, 71485487, 25796381, 28683833, 55567762, 52617748, 40452982, 16407105, 80831806, 37027059, 95197054, 82084930, 15563700, 32985356, 31708023, 66615816, 29871437, 6403680, 68853379, 60360298, 85980789, 62622988, 30757568, 8500363, 4701497, 43684289, 2109369, 27967319, 7869616, 21479207, 28646223, 84331522, 12713839, 9281126, 7650239, 64464057, 72071487, 48880195, 47221126, 76110544, 73642801, 3024223, 71981655, 63615833, 1195192, 61879611, 73815242, 25433497, 83759808, 94138403, 37948386, 44309060, 87147346, 80711112, 31552131, 10990347, 38894328]

# work dir
config['work_dir'] = "./logs/benchmarks/nyu_v2_f/pspnet_resnet50/mgda_ub_run_2"
