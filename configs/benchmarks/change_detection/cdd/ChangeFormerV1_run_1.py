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
config['model'] = {'class': models.change_detection.ChangeFormerV1, 'args': {}}

from configs.common.datasets.change_detection.train._transforms_cfg import transforms_cfg
config['train_dataset']['args']['transforms_cfg'] = transforms_cfg(size=(256, 256))
config['val_dataset']['args']['transforms_cfg'] = transforms_cfg(size=(256, 256))

# seeds
config['init_seed'] = 52377982
config['train_seeds'] = [52410969, 3137211, 69072226, 26238014, 98150551, 19287910, 18586223, 71141904, 8591368, 41567551, 74334406, 94004061, 32866580, 24271810, 44578949, 48272886, 78092346, 67387663, 35473977, 55378317, 94153854, 36524544, 34950140, 58832553, 43496583, 75144282, 69499412, 35914963, 11177090, 91123194, 81423362, 39731206, 79672227, 79040029, 78269964, 26309425, 74535021, 84474146, 59964722, 39832880, 20521224, 1081563, 99303628, 68735449, 53815048, 82203525, 3346586, 68368795, 31980160, 16128393, 73683514, 24152205, 99144767, 54723984, 96322642, 45945187, 81915923, 47890632, 59585249, 91383757, 89242156, 88081900, 6277226, 94239477, 55626187, 91543834, 75872885, 26555072, 16294235, 40772412, 48035013, 50417261, 33334329, 99326147, 81971675, 68635084, 63705598, 67891143, 7603932, 3589544, 99872093, 94333452, 35323863, 25804949, 14620732, 34345883, 95075305, 27973705, 56411261, 95609062, 13641950, 42735585, 77956496, 5576881, 25923868, 3523033, 44873082, 90411625, 3720289, 41199846]

# work dir
config['work_dir'] = "./logs/benchmarks/change_detection/cdd/ChangeFormerV1_run_1"
