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

from runners import SupervisedSingleTaskTrainer
config['runner'] = SupervisedSingleTaskTrainer

# dataset config
import data
from configs.common.datasets.change_detection.oscd import config as dataset_config
config.update(dataset_config)

# model config
from configs.common.models.change_detection.fc_siam import model_config
config['model'] = model_configconfig['model']['args']['arch'] = "FC-Siam-conc"

# optimizer config
from configs.common.optimizers.single_task_optimizer import single_task_optimizer_config
config['optimizer'] = single_task_optimizer_config

# seeds
config['init_seed'] = 65285671
config['train_seeds'] = [54757689, 69787705, 9948262, 90900580, 92218204, 8026790, 54825977, 74399507, 60992394, 11129298, 35386967, 48017264, 41131476, 10481549, 70586009, 89147692, 86190430, 30160275, 57148773, 97045218, 7378687, 53704775, 39287970, 79714884, 94345496, 91497829, 85879715, 72367563, 42236881, 84416948, 78112021, 61845979, 72941916, 21369799, 61820743, 48833343, 99504062, 99678756, 88904478, 42868361, 24025449, 97023987, 65149942, 90322087, 65951815, 17155628, 64728249, 42631895, 88291523, 25202273, 14858307, 39731685, 71061071, 60497490, 10040094, 12088343, 79097226, 85264823, 34409780, 71190308, 64012549, 19855825, 2826115, 47096495, 19867855, 13108940, 46708273, 83913038, 30957722, 41177433, 83207931, 15615925, 83536506, 31726112, 64399037, 26557735, 93895139, 92375370, 17574972, 13898692, 78251450, 99805747, 8945009, 68985382, 80647395, 39324267, 49912553, 26643642, 68279784, 32528724, 14681538, 22011885, 28711910, 22408265, 32940806, 98897216, 59668160, 95912470, 91334831, 30536218]

# work dir
config['work_dir'] = "./logs/benchmarks/change_detection/oscd/FC-Siam-conc_run_1"
