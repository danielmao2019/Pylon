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
                'labels': ['depth_estimation'],
                'meta_info': ['image_filepath', 'image_resolution'],
            },
        },
    }
dataset_config['criterion'] = dataset_config['criterion']['args']['criterion_configs']['depth_estimation']
dataset_config['metric'] = dataset_config['metric']['args']['metric_configs']['depth_estimation']
config.update(dataset_config)

# model config
from configs.common.models.nyu_v2_c.pspnet_resnet50 import model_config_depth_estimation as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers._core_ import adam_optimizer_config
config['optimizer'] = adam_optimizer_config

# seeds
config['init_seed'] = 38254849
config['train_seeds'] = [94177990, 39109888, 3882995, 15319041, 73122785, 45054435, 36324051, 67556555, 59511308, 24581306, 85850398, 19467618, 34957759, 45769703, 34821316, 38553703, 7280633, 81109523, 90489196, 49231743, 65951908, 88887492, 2937485, 85581652, 50334125, 73320877, 42703688, 51564150, 78326210, 69958009, 2450492, 21143272, 37438054, 33993779, 11866212, 64638143, 19175124, 40649398, 82635327, 47378210, 95251756, 57876257, 23604917, 14424371, 32581319, 37894541, 51916435, 72418026, 86027651, 82866288, 83689442, 32573292, 4969901, 64677755, 15371301, 66019012, 51069977, 19012442, 6241660, 15846453, 97485449, 87168161, 727348, 69362596, 42896148, 67486002, 79292883, 19615903, 6272535, 90863387, 23244163, 17848154, 23651808, 27002118, 38746627, 19880398, 88821598, 49712221, 10398259, 26318282, 69857610, 44164429, 74963278, 18820598, 62680920, 57368191, 79294055, 23325594, 35807333, 62942664, 71943173, 87120997, 19825861, 47810272, 18790738, 25991341, 20472630, 71910064, 50894250, 30149396]

# work dir
config['work_dir'] = "./logs/benchmarks/nyu_v2_c/pspnet_resnet50/single_task_depth_estimation_run_1"
