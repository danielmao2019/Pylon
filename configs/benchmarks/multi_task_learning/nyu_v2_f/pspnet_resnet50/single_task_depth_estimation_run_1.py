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
from configs.common.datasets.nyu_v2_f import config as dataset_config
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
from configs.common.models.nyu_v2_f.pspnet_resnet50 import model_config_depth_estimation as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.single_task_optimizer import single_task_optimizer_config as optimizer_config
from configs.common.optimizers.standard import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 10196745
config['train_seeds'] = [87290489, 1651066, 62886630, 72700636, 89451337, 72733118, 42842698, 48625980, 90939339, 74866094, 25059810, 71097955, 98369913, 49171607, 23327430, 51240469, 2323087, 38479753, 49522309, 25053986, 38405398, 53690444, 71579143, 32381311, 32839570, 88983882, 7247562, 86759434, 41551991, 99004822, 21141979, 47522625, 23565972, 38428901, 86685350, 79563480, 43158280, 71248934, 36339926, 32156496, 19490611, 93810125, 65825387, 66766157, 61991105, 2527924, 45695315, 44563972, 594100, 23556386, 35312199, 31811372, 55858875, 77968792, 25418787, 93346097, 75311776, 32145073, 71348455, 89891988, 21160151, 38522576, 47384465, 29959532, 9843245, 42957188, 28567053, 697420, 75299478, 75442581, 27738744, 60971319, 44626169, 58568527, 41951378, 29225869, 12039639, 9256234, 80486208, 14386108, 22857197, 1135212, 16426985, 83771782, 74837792, 36619886, 93829206, 25290217, 52920410, 69294533, 71876094, 80524188, 96348057, 38080487, 12675004, 19350175, 41542535, 38650053, 27167919, 42174523]

# work dir
config['work_dir'] = "./logs/benchmarks/nyu_v2_f/pspnet_resnet50/single_task_depth_estimation_run_1"
