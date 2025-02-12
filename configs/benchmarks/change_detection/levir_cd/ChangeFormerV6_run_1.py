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
from configs.common.datasets.change_detection.train.levir_cd import config as train_dataset_config
config.update(train_dataset_config)
from configs.common.datasets.change_detection.val.levir_cd import config as val_dataset_config
config.update(val_dataset_config)

# model config
import models
from configs.common.models.change_detection.change_former import model_config
config['model'] = model_config
config['model']['class'] = models.change_detection.ChangeFormerV6

from configs.common.datasets.change_detection._transforms_cfg import transforms_cfg
config['train_dataset']['args']['transforms_cfg'] = transforms_cfg(256)

import criteria
config['criterion']['class'] = criteria.vision_2d.ChangeFormerCriterion

# seeds
config['init_seed'] = 53780356
config['train_seeds'] = [5569849, 82328099, 5597916, 70030952, 99497390, 99098744, 74885352, 20683546, 13591608, 72191662, 3272833, 80916139, 74905206, 53928306, 79788757, 53995598, 26237029, 65838202, 99025918, 22716499, 69271253, 18269124, 26768779, 12515838, 81728971, 24750982, 83833226, 65566742, 76881702, 47732446, 64856856, 41539180, 24407878, 53097334, 24291176, 33802476, 82922494, 89971439, 60046185, 74811023, 54862724, 2466609, 18195327, 80657505, 76143272, 31970912, 75846732, 62243162, 20189395, 67282438, 63828836, 58761168, 51965258, 32266951, 564656, 98773742, 83318734, 66730480, 82727911, 84841810, 50233208, 23883315, 5986891, 94123242, 35771968, 74857225, 72198316, 76190939, 9751039, 39456232, 77809181, 46792351, 40504242, 11976976, 7297219, 55887124, 98794916, 56161024, 10241285, 5345360, 67595801, 66593584, 79132876, 7787730, 47993453, 87764662, 70888281, 44122758, 10470779, 11556410, 14974429, 40359499, 32400243, 39286558, 85160016, 69615203, 34092117, 46165755, 18549935, 27125313]

# work dir
config['work_dir'] = "./logs/benchmarks/change_detection/levir_cd/ChangeFormerV6_run_1"
