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
                'labels': ['semantic_segmentation'],
                'meta_info': ['image_filepath', 'image_resolution'],
            },
        },
    }
dataset_config['criterion'] = dataset_config['criterion']['args']['criterion_configs']['semantic_segmentation']
dataset_config['metric'] = dataset_config['metric']['args']['metric_configs']['semantic_segmentation']
config.update(dataset_config)

# model config
from configs.common.models.city_scapes_c.pspnet_resnet50 import model_config_semantic_segmentation as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers._core_ import adam_optimizer_config
config['optimizer'] = adam_optimizer_config

# seeds
config['init_seed'] = 61830725
config['train_seeds'] = [31155787, 29982711, 40326826, 79521578, 70692577, 61355227, 21524587, 74811412, 93171871, 74346244, 38438243, 68886538, 60142538, 4581724, 76938689, 53235439, 99080663, 44177490, 31694157, 50455755, 90133574, 29682210, 69691938, 99452236, 6371611, 68468618, 60302876, 64791102, 81605374, 38699575, 80390385, 55434266, 22066596, 24818777, 9140442, 10151415, 71150516, 70829758, 35274084, 62504043, 26443428, 59641889, 29047348, 73968220, 17217869, 98161023, 66109189, 34923983, 12604516, 57249028, 22451290, 58391233, 33997371, 5842284, 82155784, 19057534, 80640226, 48409816, 91420684, 61164160, 893108, 6564486, 57969252, 2855616, 98303918, 10236313, 40128241, 26359276, 45324231, 70483863, 24446923, 16419572, 11397380, 39763694, 1038765, 29630193, 91404963, 13434255, 26302441, 65794038, 69581157, 9825627, 13517785, 41466270, 13073171, 7426087, 95172529, 67129306, 52380461, 31943624, 82293824, 85333147, 96175286, 35925557, 53793653, 24558373, 6690061, 87413909, 81467500, 46731780]

# work dir
config['work_dir'] = "./logs/benchmarks/city_scapes_c/pspnet_resnet50/single_task_semantic_segmentation_run_1"