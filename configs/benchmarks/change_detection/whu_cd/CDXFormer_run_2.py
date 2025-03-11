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
from configs.common.datasets.change_detection.train.whu_cd import config as train_dataset_config
config.update(train_dataset_config)
from configs.common.datasets.change_detection.val.whu_cd import config as val_dataset_config
config.update(val_dataset_config)

# model config
import models
config['model'] = {'class': models.change_detection.CDXFormer, 'args': {}}

import criteria
config['criteria'] = {'class': criteria.vision_2d.CEDiceLoss, 'args': {}}

# seeds
config['init_seed'] = 63689065
config['train_seeds'] = [17550782, 17883833, 83793516, 69673374, 94952813, 59668527, 84452715, 13410462, 29689554, 10975249, 32072991, 52231525, 80218710, 16332147, 63231412, 44065678, 283160, 35176058, 67379713, 63946812, 35118723, 42325214, 46956530, 28322428, 55388187, 20009838, 14978155, 9166134, 65649778, 93788167, 90781098, 67829465, 29397985, 4669883, 26679955, 20793724, 39767304, 28998973, 33002, 32539188, 56402203, 87282554, 62779543, 6376244, 26303774, 43387853, 32658097, 42340432, 55323418, 57480368, 36111410, 29851532, 85511824, 20690296, 528547, 13803201, 77675801, 27911872, 2455802, 42077582, 18589140, 39373429, 17567610, 41210875, 41787573, 61534680, 29483856, 74345688, 43702940, 32514139, 30148385, 11965246, 43400931, 26409577, 15206109, 46876946, 94147068, 76873987, 25816502, 24057782, 43226658, 48213252, 96679080, 84943303, 39520914, 60313976, 47356393, 85100777, 26266111, 54667984, 8709643, 8382639, 82597244, 62820863, 62731576, 30839278, 5665410, 71068867, 12428191, 52954865]

# work dir
config['work_dir'] = "./logs/benchmarks/change_detection/whu_cd/CDXFormer_run_2"
