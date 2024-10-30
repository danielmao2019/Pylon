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
config['init_seed'] = 87830458
config['train_seeds'] = [36309790, 33833532, 39163256, 15479844, 55854936, 37863127, 44854656, 43519643, 14954578, 90118149, 30928835, 1818134, 53587073, 36640484, 60872625, 26787276, 72850541, 64317787, 93466340, 27497281, 91348769, 16142431, 52754042, 18366818, 80221257, 31185508, 47780457, 93778196, 52721412, 24820014, 75892555, 60057373, 9619539, 84008215, 44891484, 47510399, 53809677, 20803610, 47809273, 69900864, 52529518, 10673798, 38770317, 92119102, 9680944, 75005406, 47253607, 28679213, 42978650, 24643324, 28860556, 94855771, 3544752, 74624297, 35287814, 93444419, 9678382, 47333969, 1541600, 35607928, 75516144, 30291368, 26865934, 3993212, 39683620, 47338659, 49253687, 76031973, 5026430, 30102258, 30146526, 94768925, 42880004, 86892066, 96082271, 31098386, 39765117, 98090881, 14194461, 67913466, 84386206, 41788372, 38295809, 68752622, 66502799, 22780051, 23897243, 52688335, 2254976, 10762981, 73461137, 39163147, 53962289, 47210994, 47565778, 92525540, 87753116, 74657531, 73287945, 9664039]

# work dir
config['work_dir'] = "./logs/benchmarks/nyu_v2_c/pspnet_resnet50/single_task_depth_estimation_run_2"