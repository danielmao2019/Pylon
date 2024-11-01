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
from configs.common.datasets.celeb_a import config as dataset_config
config.update(dataset_config)

# model config
from configs.common.models.celeb_a.celeb_a_resnet18 import model_config_all_tasks as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.gradvac import optimizer_config
from configs.common.optimizers._core_ import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 38221812
config['train_seeds'] = [88508432, 81740614, 94396547, 66044651, 53932771, 12492217, 55301595, 41213940, 11177597, 74431855, 47774360, 60789591, 78411045, 1300476, 99357670, 70271517, 14338525, 67194527, 9867436, 99726156, 30756689, 48232835, 51030954, 76156820, 876776, 89332137, 38220929, 20490902, 41723259, 3504595, 68227972, 68835304, 94931504, 29656944, 47618674, 39816802, 73203415, 98657603, 66053778, 91041764, 89725142, 6230032, 68626706, 88678703, 59115402, 29536761, 85908451, 65418438, 13667086, 39887922, 86803305, 25548477, 3512404, 41935648, 1540703, 70356435, 71554565, 91787363, 342381, 20158596, 29858621, 56033825, 81736866, 79115409, 80319460, 23455737, 62202521, 65522938, 82934078, 90197282, 64415751, 32251876, 65259, 90209916, 73988531, 78547265, 41771550, 57048551, 52961914, 40388383, 16733864, 2961935, 29455980, 88263139, 28124944, 20758141, 63636690, 82212935, 94479416, 89582642, 31408796, 44452463, 38476843, 33158578, 77340874, 85876640, 34937832, 28010767, 94686496, 22718670]

# work dir
config['work_dir'] = "./logs/benchmarks/celeb_a/celeb_a_resnet18/gradvac_run_1"