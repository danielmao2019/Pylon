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
import models
from configs.common.models.change_detection.change_former import model_config
config['model'] = model_config
config['model']['class'] = models.change_detection.ChangeFormerV3

config['criterion']['class'] = criteria.vision_2d.ChangeFormerCriterion

# seeds
config['init_seed'] = 58471565
config['train_seeds'] = [55669956, 66069247, 79521129, 13143639, 54450693, 5448984, 98492430, 83376984, 97731698, 39831326, 23178355, 80456121, 22336659, 60757269, 35063505, 34221815, 123972, 14835219, 66693876, 78594479, 53823331, 72017286, 1502371, 78650494, 29583850, 84289919, 13315634, 89210071, 82591812, 32200814, 60066982, 19144367, 81998636, 42999931, 70999004, 77910404, 49921937, 69242023, 82995060, 37919883, 75554233, 94583976, 55923338, 84675329, 83760357, 16767403, 8924593, 97039894, 18196781, 56461203, 80061658, 38043149, 67137959, 75485253, 12332270, 66796491, 30845956, 98692744, 43194319, 47951666, 47975485, 44799842, 75153366, 58135037, 17258537, 35600796, 67048001, 11828567, 39679698, 92186897, 952855, 23544470, 73495380, 7491351, 13211093, 13064431, 70762088, 64383885, 61575081, 14350937, 61530136, 94050530, 36884290, 60899466, 97404018, 19231075, 33385330, 61326331, 50440745, 94097491, 9517583, 15528440, 3859684, 20959605, 9747600, 49846225, 71806847, 86634, 727284, 78441107]

# work dir
config['work_dir'] = "./logs/benchmarks/change_detection/cdd/ChangeFormerV3_run_0"
