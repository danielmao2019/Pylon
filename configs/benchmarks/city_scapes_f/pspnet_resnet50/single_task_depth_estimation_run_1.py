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
from configs.common.datasets.city_scapes_f import config as dataset_config
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
from configs.common.models.city_scapes_f.pspnet_resnet50 import model_config_depth_estimation as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers._core_ import adam_optimizer_config
config['optimizer'] = adam_optimizer_config

# seeds
config['init_seed'] = 17522055
config['train_seeds'] = [49633549, 51073654, 23779535, 10554953, 47523011, 19209658, 74248333, 24752518, 72017556, 40162651, 21019822, 27560668, 58639970, 89572096, 10275913, 45502089, 58463400, 79590891, 25991741, 29857999, 24240753, 4553378, 30124279, 44776230, 41403744, 38379209, 87440570, 73624596, 37276770, 64898664, 88918599, 2088994, 3887418, 75553886, 33779658, 24965801, 85476140, 93229151, 30142477, 26841212, 72392357, 51861119, 26181893, 39458396, 87333029, 34378563, 87756764, 16890727, 73440246, 6023775, 22280223, 12071016, 85092906, 76255430, 61633845, 34215385, 42401648, 58693122, 63880312, 43148471, 54973143, 59627422, 74786711, 65456345, 93439831, 69414571, 4714044, 30961094, 2178394, 91289768, 27059436, 49211990, 69282098, 49903980, 21605716, 72905816, 50540546, 7949105, 81173517, 20937849, 15194003, 68465203, 61684792, 93745964, 7120309, 42771252, 34209714, 4107550, 2586010, 77628078, 82252343, 60482781, 15694932, 9830557, 250981, 76726100, 48232052, 45877789, 8172572, 77363948]

# work dir
config['work_dir'] = "./logs/benchmarks/city_scapes_f/pspnet_resnet50/single_task_depth_estimation_run_1"
