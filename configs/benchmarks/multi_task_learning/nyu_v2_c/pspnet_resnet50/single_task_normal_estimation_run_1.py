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
from configs.common.datasets.multi_task_learning.nyu_v2_c import config as dataset_config
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
from configs.common.models.multi_task_learning.nyu_v2_c.pspnet_resnet50 import model_config_normal_estimation as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.single_task_optimizer import single_task_optimizer_config as optimizer_config
from configs.common.optimizers.standard import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 63504254
config['train_seeds'] = [68404699, 96691223, 4909211, 62517009, 62820651, 531661, 87692120, 89349553, 72908136, 93157856, 57811316, 46110822, 10430113, 421324, 87662638, 60403022, 93364728, 22457064, 61608610, 89267922, 92533419, 24137834, 54358951, 96843531, 27366757, 44935167, 41136584, 12167567, 96369450, 80247558, 3005235, 4895881, 15947143, 27591546, 8878782, 16977784, 14913799, 55435655, 45313725, 6490785, 92974731, 62195257, 81704808, 62288747, 31699084, 736458, 45956974, 7925153, 48237494, 24688023, 68747434, 46366271, 36060772, 86700419, 43706235, 78600485, 25393850, 86542150, 463053, 48246586, 51592524, 79630993, 25363324, 43270482, 53639102, 64250343, 1063784, 78880899, 90526298, 704178, 21592640, 4724286, 2292595, 240345, 29756876, 30371328, 92076551, 12257676, 36970465, 60910834, 69755795, 52277343, 65731496, 25231280, 59020453, 35701568, 52562407, 64256185, 79853524, 34907534, 77359798, 29441775, 99246223, 36432947, 13354521, 72056332, 7450786, 52874977, 23895797, 25141100]

# work dir
config['work_dir'] = "./logs/benchmarks/multi_task_learning/nyu_v2_c/pspnet_resnet50/single_task_normal_estimation_run_1"
