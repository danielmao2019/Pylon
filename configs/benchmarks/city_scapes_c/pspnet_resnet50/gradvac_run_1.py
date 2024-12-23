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
from configs.common.datasets.city_scapes_c import config as dataset_config
config.update(dataset_config)

# model config
from configs.common.models.city_scapes_c.pspnet_resnet50 import model_config_all_tasks as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.gradvac import optimizer_config
from configs.common.optimizers._core_ import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 91281314
config['train_seeds'] = [36900051, 43054877, 66894668, 84144551, 15200619, 9086747, 96024015, 38409378, 72766898, 59235397, 17069000, 36669302, 96214382, 2558693, 29592684, 24075991, 96013139, 40047656, 8876122, 9589888, 80006352, 74072990, 95569600, 11108676, 12739366, 44726314, 87953636, 11033043, 93243922, 6127340, 67790852, 3984261, 38773883, 88677204, 94022208, 29644890, 83290819, 99003018, 42337925, 81945664, 31528892, 14248120, 6695331, 34842452, 25805336, 47033153, 77053915, 61685832, 31897281, 92811064, 83212693, 93125229, 73629555, 49030960, 77484607, 45300939, 82303934, 2984095, 59530346, 96970383, 64218016, 79392380, 17091977, 69429590, 89918392, 64829134, 69122549, 5497267, 90553510, 11377139, 28928230, 59832660, 75561131, 10586527, 16198356, 58536006, 67542109, 61891502, 96966162, 19757757, 49177477, 74305939, 51690996, 61478601, 21162231, 15463492, 33528996, 28754065, 34197321, 66230316, 96089492, 5708128, 89160775, 33204725, 26862126, 7734388, 89545267, 26728384, 23015160, 29871982]

# work dir
config['work_dir'] = "./logs/benchmarks/city_scapes_c/pspnet_resnet50/gradvac_run_1"
