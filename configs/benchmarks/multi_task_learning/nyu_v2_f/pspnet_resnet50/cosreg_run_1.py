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
from configs.common.optimizers.cosreg import optimizer_config
from configs.common.optimizers.standard import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 16434411
config['train_seeds'] = [25255596, 37636351, 44790798, 69506664, 68031823, 35071093, 2807535, 57031079, 9223807, 29204875, 50937103, 99341753, 28901524, 64172172, 46436098, 55399287, 18717706, 90682925, 29150615, 45423094, 52863936, 11820467, 70739027, 61008612, 63898401, 94584447, 25900996, 3621860, 57445509, 39223006, 87043343, 79096713, 76418056, 57301307, 8336885, 24112303, 37659720, 12490562, 78721650, 16399105, 8580793, 13218760, 36721725, 86822448, 510266, 7360535, 19178269, 29251265, 15077895, 49632047, 5986513, 17177882, 60603220, 53717584, 14557307, 46658043, 42292578, 79778879, 34830703, 16858359, 67134457, 90087813, 48347866, 86184949, 15999271, 71092538, 68645620, 50860004, 34988256, 27037419, 8720021, 96781380, 43494921, 32118324, 656895, 83424672, 48316745, 84000456, 35667492, 8596671, 84870619, 76083993, 60541285, 98246685, 39932288, 9125763, 83344501, 18629662, 38921523, 22145465, 44359440, 88270039, 90151732, 50923150, 77053, 19349301, 98446614, 9970947, 23439931, 93468223]

# work dir
config['work_dir'] = "./logs/benchmarks/nyu_v2_f/pspnet_resnet50/cosreg_run_1"
