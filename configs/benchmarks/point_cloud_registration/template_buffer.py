import torch
import criteria
import metrics
import optimizers
from runners.pcr_trainers import BufferTrainer

import copy
from configs.common.datasets.point_cloud_registration.train.buffer_data_cfg import get_transforms_cfg
from configs.common.datasets.point_cloud_registration.train.buffer_data_cfg import data_cfg as train_data_cfg
from configs.common.datasets.point_cloud_registration.val.buffer_data_cfg import data_cfg as val_data_cfg
from configs.common.models.point_cloud_registration.buffer_cfg import model_cfg

optimizer_cfg = {
    'class': optimizers.SingleTaskOptimizer,
    'args': {
        'optimizer_config': {
            'class': torch.optim.Adam,
            'args': {
                'params': None,
                'lr': 1.0e-4,
                'weight_decay': 1.0e-06,
            },
        },
    },
}

scheduler_cfg = {
    'class': torch.optim.lr_scheduler.StepLR,
    'args': {
        'optimizer': None,
        'step_size': 1000,
        'gamma': 0.95,
    },
}

config = [
{
    'stage': 'Ref',
    'runner': BufferTrainer,
    'work_dir': None,
    'epochs': 100,
    # seeds
    'init_seed': None,
    'train_seeds': None,
    'val_seeds': None,
    'test_seed': None,
    # dataset config
    'train_dataset': copy.deepcopy(train_data_cfg['train_dataset']),
    'train_dataloader': copy.deepcopy(train_data_cfg['train_dataloader']),
    'criterion': {
        'class': criteria.vision_3d.point_cloud_registration.BUFFER_RefStageCriterion,
        'args': {},
    },
    'val_dataset': copy.deepcopy(val_data_cfg['val_dataset']),
    'val_dataloader': copy.deepcopy(val_data_cfg['val_dataloader']),
    'test_dataset': None,
    'test_dataloader': None,
    'metric': {
        'class': metrics.vision_3d.point_cloud_registration.BUFFER_RefStageMetric,
        'args': {},
    },
    # model config
    'model': copy.deepcopy(model_cfg),
    # optimizer config
    'optimizer': copy.deepcopy(optimizer_cfg),
    # scheduler config
    'scheduler': copy.deepcopy(scheduler_cfg),
},
{
    'stage': 'Desc',
    'runner': BufferTrainer,
    'work_dir': None,
    'epochs': 100,
    # seeds
    'init_seed': None,
    'train_seeds': None,
    'val_seeds': None,
    'test_seed': None,
    # dataset config
    'train_dataset': copy.deepcopy(train_data_cfg['train_dataset']),
    'train_dataloader': copy.deepcopy(train_data_cfg['train_dataloader']),
    'criterion': {
        'class': criteria.vision_3d.point_cloud_registration.BUFFER_DescStageCriterion,
        'args': {},
    },
    'val_dataset': copy.deepcopy(val_data_cfg['val_dataset']),
    'val_dataloader': copy.deepcopy(val_data_cfg['val_dataloader']),
    'test_dataset': None,
    'test_dataloader': None,
    'metric': {
        'class': metrics.vision_3d.point_cloud_registration.BUFFER_DescStageMetric,
        'args': {},
    },
    # model config
    'model': copy.deepcopy(model_cfg),
    # optimizer config
    'optimizer': copy.deepcopy(optimizer_cfg),
    # scheduler config
    'scheduler': copy.deepcopy(scheduler_cfg),
},
{
    'stage': 'Keypt',
    'runner': BufferTrainer,
    'work_dir': None,
    'epochs': 100,
    # seeds
    'init_seed': None,
    'train_seeds': None,
    'val_seeds': None,
    'test_seed': None,
    # dataset config
    'train_dataset': copy.deepcopy(train_data_cfg['train_dataset']),
    'train_dataloader': copy.deepcopy(train_data_cfg['train_dataloader']),
    'criterion': {
        'class': criteria.vision_3d.point_cloud_registration.BUFFER_KeyptStageCriterion,
        'args': {},
    },
    'val_dataset': copy.deepcopy(val_data_cfg['val_dataset']),
    'val_dataloader': copy.deepcopy(val_data_cfg['val_dataloader']),
    'test_dataset': None,
    'test_dataloader': None,
    'metric': {
        'class': metrics.vision_3d.point_cloud_registration.BUFFER_KeyptStageMetric,
        'args': {},
    },
    # model config
    'model': copy.deepcopy(model_cfg),
    # optimizer config
    'optimizer': copy.deepcopy(optimizer_cfg),
    # scheduler config
    'scheduler': copy.deepcopy(scheduler_cfg),
},
{
    'stage': 'Inlier',
    'runner': BufferTrainer,
    'work_dir': None,
    'epochs': 100,
    # seeds
    'init_seed': None,
    'train_seeds': None,
    'val_seeds': None,
    'test_seed': None,
    # dataset config
    'train_dataset': copy.deepcopy(train_data_cfg['train_dataset']),
    'train_dataloader': copy.deepcopy(train_data_cfg['train_dataloader']),
    'criterion': {
        'class': criteria.vision_3d.point_cloud_registration.BUFFER_InlierStageCriterion,
        'args': {},
    },
    'val_dataset': copy.deepcopy(val_data_cfg['val_dataset']),
    'val_dataloader': copy.deepcopy(val_data_cfg['val_dataloader']),
    'test_dataset': None,
    'test_dataloader': None,
    'metric': {
        'class': metrics.vision_3d.point_cloud_registration.BUFFER_InlierStageMetric,
        'args': {},
    },
    # model config
    'model': copy.deepcopy(model_cfg),
    # optimizer config
    'optimizer': copy.deepcopy(optimizer_cfg),
    # scheduler config
    'scheduler': copy.deepcopy(scheduler_cfg),
},
]

config[0]['train_dataset']['args']['transforms_cfg'] = get_transforms_cfg('Euler', 3)
config[1]['train_dataset']['args']['transforms_cfg'] = get_transforms_cfg('Euler', 1)
config[2]['train_dataset']['args']['transforms_cfg'] = get_transforms_cfg('Euler', 1)
config[3]['train_dataset']['args']['transforms_cfg'] = get_transforms_cfg('Euler', 1)

config[0]['val_dataset']['args']['transforms_cfg'] = get_transforms_cfg('Euler', 3)
config[1]['val_dataset']['args']['transforms_cfg'] = get_transforms_cfg('Euler', 1)
config[2]['val_dataset']['args']['transforms_cfg'] = get_transforms_cfg('Euler', 1)
config[3]['val_dataset']['args']['transforms_cfg'] = get_transforms_cfg('Euler', 1)

config[0]['model']['args']['config']['stage'] = 'Ref'
config[1]['model']['args']['config']['stage'] = 'Desc'
config[2]['model']['args']['config']['stage'] = 'Keypt'
config[3]['model']['args']['config']['stage'] = 'Inlier'
