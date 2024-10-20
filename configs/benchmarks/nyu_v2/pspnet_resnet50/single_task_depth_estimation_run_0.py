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
from configs.common.datasets.nyu_v2 import config as dataset_config
for key in ['train_dataset', 'val_dataset', 'test_dataset']:
    dataset_config[key] = {
        'class': data.datasets.ProjectionDatasetWrapper,
        'args': {
            'dataset_config': dataset_config[key],
            'mapping': {
                'inputs': ['image'],
                'labels': ['depth_estimation'],
                'meta_info': ['image_filepath', 'image_resolution'],
            },
        },
    }
dataset_config['criterion'] = dataset_config['criterion']['args']['criterion_configs']['depth_estimation']
dataset_config['metric'] = dataset_config['metric']['args']['metric_configs']['depth_estimation']
config.update(dataset_config)

# model config
from configs.common.models.nyu_v2.pspnet_resnet50 import model_config_depth_estimation as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers._core_ import adam_optimizer_config
config['optimizer'] = adam_optimizer_config

# seeds
config['init_seed'] = 27063254
config['train_seeds'] = [23162827, 11740430, 50618355, 92812600, 18193605, 55881703, 56781877, 61180584, 91393009, 39287914, 27293826, 52748685, 11729186, 47859618, 57063997, 68595459, 76912047, 38298549, 36559300, 28494899, 3243088, 44225300, 67074358, 16560040, 13319627, 71224093, 66072425, 27219842, 86565499, 93919445, 59801671, 98018476, 73019408, 36667430, 3685668, 59745026, 66035799, 82816140, 25110403, 90444908, 85518198, 2592808, 79046061, 83507513, 34724585, 21326960, 39655475, 6879759, 16185845, 77454151, 77535858, 73046988, 88783108, 96911671, 39202082, 35200052, 95404915, 3527981, 3155065, 12776085, 73398244, 83678933, 22210401, 60173677, 56799562, 66251930, 98163661, 55447841, 80089890, 42727380, 97701642, 37589522, 51505003, 78790477, 82807369, 17268745, 74254093, 39026604, 67657395, 70718960, 6894505, 70788833, 47064155, 4905534, 79334773, 51848275, 68816858, 60347976, 5979068, 26053481, 1252725, 64969348, 124789, 81904056, 96336639, 98672531, 16556469, 93422996, 5044355, 90365677]

# work dir
config['work_dir'] = "./logs/benchmarks/nyu_v2/pspnet_resnet50/single_task_depth_estimation_run_0"
