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
from configs.common.datasets.change_detection.train.cdd import config as train_dataset_config
config.update(train_dataset_config)
from configs.common.datasets.change_detection.val.cdd import config as val_dataset_config
config.update(val_dataset_config)

# model config
from configs.common.models.change_detection.ftn import model_config
config['model'] = model_config

import criteria
config['criterion'] = {'class': criteria.vision_2d.change_detection.FTNCriterion, 'args': {}}

# seeds
config['init_seed'] = 57352194
config['train_seeds'] = [62909950, 22456470, 81035963, 85943751, 26836583, 92370980, 65752654, 84732206, 44782179, 77966811, 51191342, 7658158, 84713606, 30522896, 56420895, 59994884, 22214649, 1843227, 75936634, 12279199, 13248326, 28505332, 96037423, 99853997, 98892150, 70779296, 20799065, 54337309, 61309045, 9121130, 82412739, 93753323, 132372, 30592701, 57537625, 16821193, 57252252, 43982426, 43993077, 97842458, 38267997, 5006184, 31462365, 10241176, 45554286, 55318267, 46013485, 90367950, 85801814, 65706970, 96911927, 20731831, 40393245, 1269024, 59221195, 41108040, 8743005, 25691116, 1936899, 13253801, 26776067, 20005380, 79498932, 72165064, 63553977, 81374961, 42913239, 59627387, 87006959, 63690357, 35333274, 94544065, 24738378, 84865243, 16749041, 93153042, 39471074, 79334501, 36971152, 63528055, 46515260, 27783358, 86989205, 68910366, 8787812, 94763006, 48690346, 47193065, 66511414, 17014839, 39919285, 16895825, 21819237, 42685001, 90815326, 80925130, 28062436, 78703650, 97058552, 98688388]

# work dir
config['work_dir'] = "./logs/benchmarks/change_detection/cdd/FTN_run_2"
