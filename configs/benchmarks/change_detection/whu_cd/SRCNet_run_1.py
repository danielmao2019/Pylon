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
config['model'] = {'class': models.change_detection.SRCNet, 'args': {}}

# criterion config
import criteria
config['criterion'] = {'class': criteria.vision_2d.change_detection.SRCNetCriterion, 'args': {}}

# seeds
config['init_seed'] = 40458165
config['train_seeds'] = [81115949, 66145973, 45995882, 24936610, 960457, 27995903, 94084622, 70465156, 45649091, 31548707, 82727459, 83420898, 25551389, 31134939, 15817676, 29384815, 7681490, 53770843, 45189322, 15890538, 74556430, 59319996, 16584221, 34992725, 17326096, 95193982, 35034268, 36582462, 15508046, 13350729, 20340095, 19099364, 59776808, 51485562, 32091606, 53613315, 32035134, 985326, 98829310, 76154295, 33547796, 80986873, 19253494, 84010312, 62638034, 28244740, 61263424, 32823126, 41542142, 14456404, 79932719, 20512249, 74545309, 82317723, 13558440, 20258973, 33458709, 23632927, 53901834, 65915264, 40517214, 19531821, 28606085, 61134193, 28183220, 71427202, 93769743, 40606754, 19929807, 54009690, 68339357, 2706691, 46807135, 99720521, 71691358, 67226875, 57389067, 32268785, 98185268, 91758466, 92191891, 42385783, 6281971, 4772765, 57200001, 37285352, 68590254, 41990479, 20841855, 94605997, 47370596, 43660374, 73088156, 69417136, 3528457, 66694201, 36622662, 62952576, 30148258, 384023]

# work dir
config['work_dir'] = "./logs/benchmarks/change_detection/whu_cd/SRCNet_run_1"
