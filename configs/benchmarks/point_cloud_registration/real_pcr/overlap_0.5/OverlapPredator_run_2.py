# This file is automatically generated by `./configs/benchmarks/point_cloud_registration/gen.py`.
# Please do not attempt to modify manually.
import torch
import optimizers


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
    'criterion': None,
    'val_dataset': None,
    'val_dataloader': None,
    'test_dataset': None,
    'test_dataloader': None,
    'metric': None,
    # model config
    'model': None,
    # optimizer config
    'optimizer': {
        'class': optimizers.SingleTaskOptimizer,
        'args': {
            'optimizer_config': {
                'class': torch.optim.Adam,
                'args': {
                    'params': None,
                    'lr': 1.0e-4,
                    'weight_decay': 1.0e-06,
                },
            },
        },
    },
    # scheduler config
    'scheduler': {
        'class': torch.optim.lr_scheduler.StepLR,
        'args': {
            'optimizer': None,
            'step_size': 1000,
            'gamma': 0.95,
        },
    },
}

from runners import SupervisedSingleTaskTrainer
config['runner'] = SupervisedSingleTaskTrainer

# data config
from configs.common.datasets.point_cloud_registration.train.overlappredator_real_pcr_data_cfg import data_cfg as train_data_cfg
train_data_cfg['train_dataset']['args']['overlap'] = 0.5
config.update(train_data_cfg)
from configs.common.datasets.point_cloud_registration.val.overlappredator_real_pcr_data_cfg import data_cfg as val_data_cfg
val_data_cfg['val_dataset']['args']['overlap'] = 0.5
config.update(val_data_cfg)

# model config
from configs.common.models.point_cloud_registration.overlappredator_cfg import model_cfg
config['model'] = model_cfg

from configs.common.criteria.point_cloud_registration.overlappredator_criterion_cfg import criterion_cfg
config['criterion'] = criterion_cfg

from configs.common.metrics.point_cloud_registration.overlappredator_metric_cfg import metric_cfg
config['metric'] = metric_cfg

# seeds
config['init_seed'] = 34462866
config['train_seeds'] = [49095616, 10849405, 60378940, 39288660, 7360295, 27102282, 91223815, 28797925, 35516014, 32665031, 24429179, 17227946, 92297474, 52177563, 18482659, 75393309, 54095188, 63497973, 86752113, 46174095, 61901157, 46785043, 47456294, 68756632, 69704040, 8324826, 59275169, 86205759, 3759088, 34932466, 816007, 50811230, 18488027, 66768308, 77902201, 48300521, 85157149, 97142410, 55195333, 94236323, 82729382, 47600585, 24877403, 91777949, 75030541, 29200991, 305522, 76145546, 29686570, 64193288, 81865567, 35493922, 80813782, 71600654, 27754534, 46034767, 77290577, 14871776, 55108138, 75704902, 20287428, 48643682, 19462190, 54232336, 43339683, 48234494, 74199657, 13896209, 90254392, 75951089, 84695502, 99520294, 97965624, 52742364, 66306174, 64786090, 54971150, 18787784, 57413601, 31559503, 54087554, 23892247, 75202596, 7002024, 69913448, 52866749, 15682574, 64100062, 8943725, 80673686, 26786191, 61423722, 15780180, 73094408, 85941760, 97651303, 89040882, 70577273, 73212610, 58725358]

# work dir
config['work_dir'] = "./logs/benchmarks/point_cloud_registration/real_pcr/overlap_0.5/OverlapPredator_run_2"
