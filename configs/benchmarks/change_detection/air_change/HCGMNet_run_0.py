# This file is automatically generated by `./configs/benchmarks/change_detection/gen_bi_temporal.py`.
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
                'class': torch.optim.SGD,
                'args': {
                    'lr': 1.0e-03,
                    'momentum': 0.9,
                    'weight_decay': 1.0e-04,
                },
            },
        },
    },
    # scheduler config
    'scheduler': {
        'class': torch.optim.lr_scheduler.PolynomialLR,
        'args': {
            'optimizer': None,
            'total_iters': None,
            'power': 0.9,
        },
    },
}

from runners import SupervisedSingleTaskTrainer
config['runner'] = SupervisedSingleTaskTrainer

# dataset config
from configs.common.datasets.change_detection.train.air_change import config as train_dataset_config
config.update(train_dataset_config)
from configs.common.datasets.change_detection.val.air_change import config as val_dataset_config
config.update(val_dataset_config)

# model config
import models
config['model'] = {'class': models.change_detection.HCGMNet, 'args': {}}

# criterion config
import criteria
config['criterion'] = {
    'class': criteria.wrappers.AuxiliaryOutputsCriterion,
    'args': {
        'criterion_cfg': config['criterion'],
        'reduction': 'sum',
    },
}

# seeds
config['init_seed'] = 35233616
config['train_seeds'] = [2467932, 71653508, 89038934, 28225460, 23157678, 44873194, 60964475, 89388874, 14816166, 54832520, 8165093, 33277760, 61635259, 49444977, 55482934, 88964142, 10918862, 86306381, 94082857, 3655781, 75511618, 24204803, 57153159, 19273340, 18783255, 79741505, 78086081, 66754161, 43648997, 87658878, 72574246, 83178250, 54136013, 50182130, 19414550, 93566054, 90212863, 52526555, 89900632, 14504956, 86364384, 53767932, 56235194, 13983974, 33371009, 69045602, 16449363, 38450798, 5206622, 31729985, 45651219, 82516209, 54716703, 96391213, 60168952, 59719072, 34212020, 58873771, 39201199, 43222886, 95996563, 29722477, 6812576, 92803483, 60056783, 67771344, 26278648, 41162651, 37167230, 2201110, 12838479, 92462845, 7050295, 81027616, 94908565, 1354572, 18106173, 336908, 47673755, 9255367, 2898103, 51159023, 66090006, 22432581, 40293511, 99584499, 8827099, 91630051, 15314678, 81311807, 31369369, 45681387, 53856251, 41778663, 21464230, 29790837, 12720176, 97490224, 39410653, 89664105]

# work dir
config['work_dir'] = "./logs/benchmarks/change_detection/air_change/HCGMNet_run_0"
