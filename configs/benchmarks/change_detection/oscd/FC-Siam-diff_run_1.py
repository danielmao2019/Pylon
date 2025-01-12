# This file is automatically generated by `./configs/benchmarks/change_detection/gen.py`.
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
from configs.common.datasets.change_detection.oscd import config as dataset_config
config.update(dataset_config)

# model config
from configs.common.models.change_detection.oscd.fc_siam import model_config
config['model'] = model_config
config['model']['args']['arch'] = "FC-Siam-diff"

# optimizer config
from configs.common.optimizers.single_task_optimizer import single_task_optimizer_config
config['optimizer'] = single_task_optimizer_config

# seeds
config['init_seed'] = 15172937
config['train_seeds'] = [74722224, 57810875, 79620427, 86043468, 98327508, 94661591, 85307516, 54518915, 82641154, 70231809, 42623462, 8555953, 20469308, 75043660, 87045228, 74532541, 50906481, 69145015, 75657789, 71992392, 13794653, 51375778, 32319856, 98087441, 13274011, 7947790, 52232399, 57091020, 96498018, 90519949, 51837147, 25503237, 95489770, 93243046, 51091656, 68041806, 46037016, 74445856, 69160801, 30640261, 5425336, 71653613, 80262203, 13355709, 93594874, 2538447, 17280032, 56718568, 89180447, 69805312, 19142051, 24119085, 49789706, 4666323, 5145810, 6469686, 34181901, 25492845, 51035983, 10089337, 96702544, 23625131, 80589735, 99867474, 8491251, 95036434, 35522706, 38430848, 66637979, 13583746, 23729040, 70354553, 14098252, 91548551, 27087422, 26248289, 5304214, 59148590, 32784846, 94543830, 3741446, 51245527, 92994486, 38363335, 10644429, 23720102, 24096541, 41022026, 94405386, 47985449, 14859419, 41035982, 18739726, 65379925, 26558771, 31275254, 46102273, 42232565, 19649960, 17431334]

# work dir
config['work_dir'] = "./logs/benchmarks/change_detection/oscd/FC-Siam-diff_run_1"
