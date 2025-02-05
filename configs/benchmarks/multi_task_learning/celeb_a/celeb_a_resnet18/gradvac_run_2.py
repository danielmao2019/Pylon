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
from configs.common.datasets.multi_task_learning.celeb_a import config as dataset_config
config.update(dataset_config)

# model config
from configs.common.models.multi_task_learning.celeb_a.celeb_a_resnet18 import model_config_all_tasks as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.multi_task_learning.gradvac import optimizer_config
from configs.common.optimizers.standard import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 75616594
config['train_seeds'] = [50155456, 20739956, 80723387, 42960353, 28764832, 47314623, 89586682, 80608476, 15054750, 12172812, 80644963, 50586372, 83150966, 79172207, 86583313, 60736309, 57739871, 67320073, 50871928, 66266671, 99487410, 4710497, 23790678, 98196802, 94461372, 76239158, 64310141, 78626863, 75750632, 73842004, 80672480, 18697180, 19274860, 44528490, 58571203, 38609084, 29916391, 48202274, 24095547, 74176508, 7530518, 31989931, 79336384, 53789510, 13022525, 98329886, 13665549, 16069089, 31048956, 95719473, 51156174, 62710870, 6150531, 33218902, 28207221, 59456243, 5448759, 59568245, 1861762, 98124988, 9400032, 75066439, 29115378, 92298158, 85170577, 66252772, 43889762, 29055059, 53814292, 68111120, 71928876, 3223569, 31153258, 46070366, 88533541, 31047446, 29447363, 46177609, 71034132, 28093460, 94295524, 55221046, 51385193, 76017809, 6965338, 57533011, 80102525, 62313804, 11830303, 82826455, 9590878, 74348336, 35703995, 71144251, 47049559, 38104784, 86234204, 88491319, 29091375, 81403330]

# work dir
config['work_dir'] = "./logs/benchmarks/multi_task_learning/celeb_a/celeb_a_resnet18/gradvac_run_2"
