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
                'labels': ['semantic_segmentation'],
                'meta_info': ['image_filepath', 'image_resolution'],
            },
        },
    }
dataset_config['criterion'] = dataset_config['criterion']['args']['criterion_configs']['semantic_segmentation']
dataset_config['metric'] = dataset_config['metric']['args']['metric_configs']['semantic_segmentation']
config.update(dataset_config)

# model config
from configs.common.models.city_scapes_f.pspnet_resnet50 import model_config_semantic_segmentation as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers._core_ import adam_optimizer_config
config['optimizer'] = adam_optimizer_config

# seeds
config['init_seed'] = 38734150
config['train_seeds'] = [28525204, 93546878, 19344298, 59739112, 42693816, 7598292, 33130044, 59287658, 99901380, 59475062, 50210455, 56519328, 72055748, 19459510, 9864452, 43633934, 75679565, 43864379, 7364271, 34971497, 83928785, 42714722, 67652796, 5236624, 16129760, 85686350, 52446144, 76251583, 8162801, 4443550, 36008240, 19411578, 96179949, 8094118, 37448191, 51822316, 81115371, 22439237, 74555585, 85047980, 67032608, 87583849, 58720276, 98439149, 701297, 70315366, 27105214, 33815215, 12302584, 51901008, 65681696, 93502133, 82066009, 28437408, 58855662, 40330268, 50314546, 82032114, 53855455, 26437302, 40147243, 30895151, 911118, 50154139, 24582983, 83275283, 27096309, 3014540, 21659644, 4036464, 31966575, 13809978, 81615733, 95308567, 13075548, 26947706, 95548197, 45828574, 23176481, 3704019, 52687634, 80218970, 49798761, 39206395, 30700769, 32691520, 57362815, 81858844, 90203, 62161594, 93434957, 69677966, 85618569, 84610640, 30863204, 11521388, 8998258, 56435917, 13913379, 42884685]

# work dir
config['work_dir'] = "./logs/benchmarks/city_scapes_f/pspnet_resnet50/single_task_semantic_segmentation_run_0"