# This file is automatically generated by `./configs/benchmarks/change_detection/gen_sysu_cd.py`.
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

from runners import SupervisedSingleTaskTrainer
config['runner'] = SupervisedSingleTaskTrainer

# dataset config
from configs.common.datasets.change_detection.i3pe_sysu_cd import config as dataset_config
config.update(dataset_config)

# model config
from configs.common.models.change_detection.sysu_cd.i3pe import model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.single_task_optimizer import single_task_optimizer_config as optimizer_config
from configs.common.optimizers._core_ import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 57272912
config['train_seeds'] = [76498368, 77505994, 73630834, 12802908, 48227167, 82888916, 78851137, 87653369, 25360146, 14215337, 73507610, 61729145, 73259191, 31649919, 20592225, 57530484, 30871878, 38619005, 49519696, 49393162, 66800767, 88298105, 62494741, 43496034, 51969256, 58939476, 82508568, 91441552, 4553880, 30655717, 38136306, 14173133, 59262453, 27990437, 98038214, 70159092, 91175697, 85413732, 28309682, 16953411, 92361256, 57736476, 65376094, 62159984, 19169418, 18733378, 5594040, 17139208, 42857672, 12228500, 74529678, 26249899, 76948853, 96165431, 11935736, 98309603, 14723822, 63195719, 90971514, 25604532, 16846312, 71733922, 86417144, 70700762, 78488454, 35727120, 86070105, 19927450, 58306732, 81604734, 91817880, 89164157, 21374037, 56578855, 23998037, 77479083, 75595411, 11187473, 1359834, 44872288, 38214313, 43311868, 77088330, 29838167, 58717072, 41689363, 34227799, 45061362, 5644779, 41194819, 82890616, 5509007, 45830986, 55339087, 72082046, 76046841, 50498559, 67314644, 7054856, 63979652]

# work dir
config['work_dir'] = "./logs/benchmarks/change_detection/sysu_cd/i3pe_run_0"
