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
from configs.common.models.change_detection.snunet import model_config
config['model'] = model_config
import criteria
config['criterion'] = {'class': criteria.vision_2d.change_detection.SNUNetCDCriterion, 'args': {}}
 
# seeds
config['init_seed'] = 53713097
config['train_seeds'] = [34279070, 53897920, 39693928, 4336976, 23803824, 10989567, 6682143, 41579915, 25445726, 81147502, 77326752, 34980779, 98542483, 68289203, 95000049, 64791564, 53609135, 72950139, 47652593, 68106224, 31083443, 23582177, 97807756, 40887431, 26929684, 12527722, 85887356, 16619112, 95675069, 80356914, 30240040, 28243334, 28801940, 52935593, 18373517, 22923944, 42016324, 81497797, 85879767, 54334668, 11849571, 39963528, 43582040, 37056473, 16725975, 76630881, 15929962, 3725021, 53125722, 97397116, 49977761, 1681067, 87347704, 7250933, 62432084, 13591622, 12865705, 19923582, 6757651, 21044509, 379509, 10928987, 13759673, 85435076, 62032030, 80584826, 49064666, 50385637, 92630429, 93190288, 48763125, 86051920, 75909114, 53541151, 51457900, 30933015, 77847328, 15841309, 37704240, 65421971, 77019999, 34464752, 34855464, 77116366, 52914035, 49529852, 8267001, 27276001, 35340472, 4889919, 42288603, 39038999, 7807946, 56698802, 6935734, 25071370, 15821461, 19395349, 85140282, 35062294]

# work dir
config['work_dir'] = "./logs/benchmarks/change_detection/levir_cd/SNUNet_ECAM_run_0"
