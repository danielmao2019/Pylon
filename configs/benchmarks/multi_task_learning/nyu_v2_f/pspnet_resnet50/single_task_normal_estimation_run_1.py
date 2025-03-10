# This file is automatically generated by `./configs/benchmarks/multi_task_learning/gen_multi_task_learning.py`.
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
from configs.common.datasets.multi_task_learning.nyu_v2_f import config as dataset_config
for key in ['train_dataset', 'val_dataset', 'test_dataset']:
    dataset_config[key] = {
        'class': data.datasets.ProjectionDatasetWrapper,
        'args': {
            'dataset_config': dataset_config[key],
            'mapping': {
                'inputs': ['image'],
                'labels': ['normal_estimation'],
                'meta_info': ['image_filepath', 'image_resolution'],
            },
        },
    }
dataset_config['criterion'] = dataset_config['criterion']['args']['criterion_configs']['normal_estimation']
dataset_config['metric'] = dataset_config['metric']['args']['metric_configs']['normal_estimation']
config.update(dataset_config)

# model config
from configs.common.models.multi_task_learning.nyu_v2_f.pspnet_resnet50 import model_config_normal_estimation as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.single_task_optimizer import single_task_optimizer_config as optimizer_config
from configs.common.optimizers.standard import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 38978634
config['train_seeds'] = [50266405, 35612477, 8971469, 35309899, 40622109, 60237788, 18931221, 72176456, 93523383, 99473934, 93308916, 4032894, 41256254, 3026513, 52953687, 21020121, 43157606, 21917775, 73332825, 69863728, 3146457, 14719128, 28264833, 39609308, 23087796, 33601923, 81345990, 98286976, 6856750, 75268896, 62796030, 16906222, 13480209, 78077259, 26896694, 70686008, 26289886, 77765604, 28949515, 37326312, 65468745, 30117317, 48787237, 87355651, 13359600, 65075993, 73966593, 40398909, 94605163, 8763875, 90924500, 54041334, 32315420, 10236711, 36884429, 1661710, 37083400, 16677638, 1762647, 27527180, 52550573, 63826172, 6669230, 24180892, 13430060, 2998740, 6020690, 30893020, 92446284, 19448480, 80301172, 94727850, 59624583, 4243261, 7579844, 76907161, 38148586, 3684990, 49594099, 99067374, 81468768, 2085763, 58110707, 85172356, 62618678, 14379713, 10793425, 82037785, 30115868, 50121281, 21619289, 26414942, 81676949, 39317061, 18559290, 67008731, 2291741, 3393835, 34189581, 71503581]

# work dir
config['work_dir'] = "./logs/benchmarks/multi_task_learning/nyu_v2_f/pspnet_resnet50/single_task_normal_estimation_run_1"
