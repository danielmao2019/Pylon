# This file is automatically generated by `./configs/benchmarks/multi_task_learning/gen_multi_task_learning.py`.
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
from configs.common.datasets.multi_task_learning.city_scapes_f import config as dataset_config
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
from configs.common.models.multi_task_learning.city_scapes_f.pspnet_resnet50 import model_config_depth_estimation as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.single_task_optimizer import single_task_optimizer_config as optimizer_config
from configs.common.optimizers.standard import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 47762597
config['train_seeds'] = [85188044, 90146725, 69246072, 60818369, 64075852, 54811852, 35262932, 99629019, 8487178, 30442589, 91143851, 47318143, 48500161, 49668070, 55625108, 17257659, 54146346, 46638650, 7107042, 82984492, 91381263, 76723797, 5961856, 26795950, 37818204, 87293871, 93026975, 24371651, 94172702, 6331409, 56537766, 21364307, 5093287, 67089866, 10076476, 42343976, 38813702, 41297596, 79356448, 88459775, 66113857, 65124698, 46519038, 20480998, 53523225, 12611456, 48458874, 55538786, 89644490, 6043291, 25678788, 40322544, 72155276, 60895041, 35913758, 68057837, 41615543, 15180752, 17270151, 55051435, 44835997, 47591625, 29666494, 82582183, 66221923, 58788728, 22323939, 71043087, 8822675, 27247245, 57788900, 42754810, 46320202, 64158045, 25808996, 75504425, 23471102, 82355384, 30711880, 63792821, 50958420, 15662451, 24924125, 41218022, 54306390, 20771234, 13051558, 83996350, 75309643, 84259329, 12185441, 35713840, 92329194, 4106200, 90933028, 77565991, 17115500, 12436549, 41140932, 76798755]

# work dir
config['work_dir'] = "./logs/benchmarks/multi_task_learning/city_scapes_f/pspnet_resnet50/single_task_depth_estimation_run_0"
