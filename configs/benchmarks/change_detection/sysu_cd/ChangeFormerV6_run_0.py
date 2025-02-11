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
from configs.common.datasets.change_detection.train.sysu_cd import config as train_dataset_config
config.update(train_dataset_config)
from configs.common.datasets.change_detection.val.sysu_cd import config as val_dataset_config
config.update(val_dataset_config)

# model config
from configs.common.models.change_detection.change_former import change_former_v6_config as model_config
config['model'] = model_config

# seeds
config['init_seed'] = 57272912
config['train_seeds'] = [76498368, 77505994, 73630834, 12802908, 48227167, 82888916, 78851137, 87653369, 25360146, 14215337, 73507610, 61729145, 73259191, 31649919, 20592225, 57530484, 30871878, 38619005, 49519696, 49393162, 66800767, 88298105, 62494741, 43496034, 51969256, 58939476, 82508568, 91441552, 4553880, 30655717, 38136306, 14173133, 59262453, 27990437, 98038214, 70159092, 91175697, 85413732, 28309682, 16953411, 92361256, 57736476, 65376094, 62159984, 19169418, 18733378, 5594040, 17139208, 42857672, 12228500, 74529678, 26249899, 76948853, 96165431, 11935736, 98309603, 14723822, 63195719, 90971514, 25604532, 16846312, 71733922, 86417144, 70700762, 78488454, 35727120, 86070105, 19927450, 58306732, 81604734, 91817880, 89164157, 21374037, 56578855, 23998037, 77479083, 75595411, 11187473, 1359834, 44872288, 38214313, 43311868, 77088330, 29838167, 58717072, 41689363, 34227799, 45061362, 5644779, 41194819, 82890616, 5509007, 45830986, 55339087, 72082046, 76046841, 50498559, 67314644, 7054856, 63979652]

# work dir
config['work_dir'] = "./logs/benchmarks/change_detection/sysu_cd/ChangeFormerV6_run_0"
