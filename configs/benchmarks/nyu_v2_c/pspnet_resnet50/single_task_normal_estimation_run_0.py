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
from configs.common.optimizers.single_task_optimizer import single_task_optimizer_config as optimizer_config
from configs.common.optimizers._core_ import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 81875336
config['train_seeds'] = [97060213, 80888769, 1447305, 71573306, 8487308, 61267238, 83578678, 55752962, 40369490, 87235675, 61904364, 9650511, 36876976, 88494269, 39611755, 14405399, 92396118, 26979977, 5015331, 61103480, 28385957, 27076957, 89682695, 49467299, 26608752, 36036858, 44219005, 37821335, 58731922, 45999718, 96527964, 43185921, 33114379, 4100273, 51854809, 44327866, 941840, 46311656, 68356219, 44562678, 67036370, 53795476, 56019321, 24519447, 88735583, 44164625, 17603324, 43874026, 59132770, 60010540, 92182989, 86862074, 99485105, 30777356, 54037485, 71317013, 326921, 83687376, 79540382, 29458424, 73218911, 55315464, 39198525, 81718716, 35888062, 92094468, 68194437, 58064550, 59702632, 26367295, 17941036, 15394622, 29778989, 12742558, 10345625, 42892991, 69462460, 4744832, 83122335, 81044177, 91509795, 30366689, 34987381, 94603889, 82008758, 26913905, 15651, 11459738, 11781738, 62036284, 40479901, 13150369, 40059801, 37639413, 17355053, 10951513, 6188977, 79245086, 54832106, 13647759]

# work dir
config['work_dir'] = "./logs/benchmarks/nyu_v2_c/pspnet_resnet50/single_task_normal_estimation_run_0"
