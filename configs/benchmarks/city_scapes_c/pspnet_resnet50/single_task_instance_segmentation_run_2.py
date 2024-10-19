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
from configs.common.datasets.city_scapes_c import config as dataset_config
for key in ['train_dataset', 'val_dataset', 'test_dataset']:
    dataset_config[key] = {
        'class': data.datasets.ProjectionDatasetWrapper,
        'args': {
            'dataset_config': dataset_config[key],
            'mapping': {
                'inputs': ['image'],
                'labels': ['instance_segmentation'],
                'meta_info': ['image_filepath', 'image_resolution'],
            },
        },
    }
dataset_config['criterion'] = dataset_config['criterion']['args']['criterion_configs']['instance_segmentation']
dataset_config['metric'] = dataset_config['metric']['args']['metric_configs']['instance_segmentation']
config.update(dataset_config)

# model config
from configs.common.models.city_scapes_c.pspnet_resnet50 import model_config_instance_segmentation as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers._core_ import adam_optimizer_config
config['optimizer'] = adam_optimizer_config

# seeds
config['init_seed'] = 63230493
config['train_seeds'] = [66923501, 40422865, 38167847, 73772191, 73021709, 83625927, 74988533, 83399851, 89764430, 70420012, 98516590, 22009075, 86757381, 75770443, 65128688, 55609845, 57326934, 74535038, 74824933, 99125081, 45823304, 80024145, 55013557, 31023368, 56497974, 3010279, 45784304, 25506644, 6420485, 69109781, 7851310, 17007811, 66993960, 14706195, 58578190, 80488408, 2992366, 59823329, 72700574, 99303424, 2937955, 25840629, 64937897, 45872550, 64044608, 52941617, 47161497, 4951073, 30229216, 77461930, 76760642, 71503228, 10063759, 58181870, 30191297, 48071213, 73704800, 61124011, 67422230, 27617744, 42348170, 99733561, 19271316, 25768647, 31142315, 29971178, 99571740, 35290877, 91355880, 1696491, 97942698, 40265137, 49491379, 62599405, 7831477, 74743864, 2865444, 68332869, 67973391, 61268754, 85659412, 10873982, 4946969, 29202281, 22157329, 20965626, 23934584, 92608903, 38643648, 36408733, 59935716, 4602312, 79734530, 7027172, 7646975, 70566390, 64426138, 72335499, 61437901, 148594]

# work dir
config['work_dir'] = "./logs/benchmarks/city_scapes_c/pspnet_resnet50/single_task_instance_segmentation_run_2"
