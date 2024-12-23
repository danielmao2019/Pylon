# This file is automatically generated by `./configs/benchmarks/multi_task_learning.py`.
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
from configs.common.datasets.city_scapes_f import config as dataset_config
config.update(dataset_config)

# model config
from configs.common.models.city_scapes_f.pspnet_resnet50 import model_config_all_tasks as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.graddrop import optimizer_config
from configs.common.optimizers._core_ import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 51648731
config['train_seeds'] = [4442286, 97062643, 18237100, 40891400, 9815558, 68172250, 87349946, 5685906, 25205620, 17235805, 66732463, 92468549, 2640417, 15551341, 22249208, 78236900, 30042281, 17289285, 89395253, 49725593, 6665350, 96225389, 30623101, 46188888, 33391469, 57987022, 21851189, 70136819, 56581072, 81071831, 84386487, 2170745, 38895500, 6985930, 23514224, 87133015, 47970024, 34865536, 67368391, 89240302, 92748889, 34489208, 31259806, 35750018, 3699237, 97619605, 1192344, 96753849, 8198817, 80173760, 21580828, 68822504, 33739029, 2674772, 30278375, 48239925, 67294329, 97799866, 22832768, 66373327, 98938000, 99622374, 62914716, 20966544, 83670871, 67803321, 98334993, 81235291, 37390275, 89013079, 7017611, 59971936, 63166977, 53936264, 92857405, 67799230, 4943899, 1949009, 87284341, 55404609, 521363, 15277221, 99225659, 44772951, 37231760, 52632684, 1979289, 15513577, 85089584, 33542500, 96519367, 15112629, 6733580, 88968966, 16748818, 55429566, 72884752, 34955408, 24786473, 20564026]

# work dir
config['work_dir'] = "./logs/benchmarks/city_scapes_f/pspnet_resnet50/graddrop_run_1"
