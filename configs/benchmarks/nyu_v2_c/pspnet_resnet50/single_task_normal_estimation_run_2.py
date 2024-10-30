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
from configs.common.datasets.nyu_v2_c import config as dataset_config
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
from configs.common.models.nyu_v2_c.pspnet_resnet50 import model_config_normal_estimation as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers._core_ import adam_optimizer_config
config['optimizer'] = adam_optimizer_config

# seeds
config['init_seed'] = 49649211
config['train_seeds'] = [61140842, 85624023, 67349933, 62449745, 67417707, 8659082, 86244309, 89827017, 16921715, 45134630, 41578762, 34061580, 52622903, 98990067, 46049383, 41939416, 40507375, 75543318, 23862625, 21197341, 5714715, 77474350, 84141618, 48400178, 77097280, 50517800, 76490252, 97747841, 45840248, 6689572, 49572582, 83326008, 68000357, 20899948, 27237070, 55355593, 56444209, 95480008, 17705612, 7305739, 34644339, 23497408, 76698809, 43126684, 12453848, 91175569, 2381651, 59445513, 48583109, 24088702, 1070958, 2274488, 61066387, 64090418, 54132719, 27089639, 87137137, 72447348, 39135774, 98425000, 82916125, 29494369, 21701981, 27079191, 45357301, 3820430, 59171923, 16001656, 28650515, 46497111, 95540510, 823771, 22077218, 98413602, 96677410, 89777841, 70630538, 50557896, 88096798, 62477067, 76318093, 82422897, 95884764, 16639794, 55809133, 47446789, 83539403, 53337078, 63333764, 50621849, 59890414, 56279994, 67859185, 71244952, 43292400, 57963380, 68793143, 74836890, 51640181, 69254546]

# work dir
config['work_dir'] = "./logs/benchmarks/nyu_v2_c/pspnet_resnet50/single_task_normal_estimation_run_2"