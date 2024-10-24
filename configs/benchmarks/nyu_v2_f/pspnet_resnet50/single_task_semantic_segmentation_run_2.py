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
                'labels': ['semantic_segmentation'],
                'meta_info': ['image_filepath', 'image_resolution'],
            },
        },
    }
dataset_config['criterion'] = dataset_config['criterion']['args']['criterion_configs']['semantic_segmentation']
dataset_config['metric'] = dataset_config['metric']['args']['metric_configs']['semantic_segmentation']
config.update(dataset_config)

# model config
from configs.common.models.nyu_v2_f.pspnet_resnet50 import model_config_semantic_segmentation as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers._core_ import adam_optimizer_config
config['optimizer'] = adam_optimizer_config

# seeds
config['init_seed'] = 70564668
config['train_seeds'] = [35477605, 78671627, 74967175, 5927490, 37574215, 76699722, 1699830, 77787261, 22318622, 63416728, 51812932, 73176952, 1115370, 49206205, 53225413, 6138147, 29105703, 73121792, 61295745, 42047555, 65497997, 69425057, 52354734, 43146408, 28377059, 53644741, 1957049, 41262981, 99125126, 28360237, 41656011, 47183265, 26650065, 56687150, 95189249, 87441461, 56331815, 2790741, 1962372, 25217346, 37590936, 2023506, 58169569, 52550129, 13268584, 63213099, 61949122, 49201860, 6357134, 30766030, 43352478, 46169630, 90490600, 5416173, 48983170, 19428677, 86446026, 51748674, 75649699, 89323827, 64661326, 27664560, 80732478, 21060828, 48535504, 27628771, 77183531, 67740406, 25596522, 75515243, 4326247, 71712124, 98688945, 93918191, 45816656, 10205267, 78428935, 85241473, 26374179, 30149809, 28631476, 10579360, 18249602, 15145647, 4603619, 80637445, 36341925, 11969676, 10761513, 14762961, 59737066, 42182341, 15703624, 10226932, 3570082, 30926636, 84166963, 9303459, 96363192, 2045389]

# work dir
config['work_dir'] = "./logs/benchmarks/nyu_v2_f/pspnet_resnet50/single_task_semantic_segmentation_run_2"
