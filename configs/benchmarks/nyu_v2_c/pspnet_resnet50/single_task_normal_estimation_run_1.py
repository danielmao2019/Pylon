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
from configs.common.optimizers._core_ import adam_optimizer_config
config['optimizer'] = adam_optimizer_config

# seeds
config['init_seed'] = 79231116
config['train_seeds'] = [62807632, 80678087, 33795624, 48978210, 12585856, 19607533, 20857635, 96794481, 28150236, 74943511, 58247050, 77813155, 29839908, 91473096, 33648708, 67851307, 74608790, 14291918, 74157916, 98270461, 33396651, 17772762, 4296340, 300259, 566745, 31984384, 65043388, 62668034, 46318539, 61455391, 32040987, 89696961, 22207830, 16108934, 33684569, 13284199, 73371143, 25407122, 31175786, 24462141, 78950672, 80140171, 86624340, 54491271, 47602794, 62732153, 70851089, 12085087, 19377523, 89488823, 44067071, 44790993, 1813938, 417264, 87835982, 66500531, 65916929, 48343225, 85878339, 33104386, 3350301, 55706385, 44286849, 45953485, 44251405, 79855140, 55287908, 24702396, 78809513, 9200035, 87881296, 58071337, 37218604, 19432934, 95941779, 35727990, 29634023, 5503437, 16997318, 28034699, 91609128, 26799765, 87342731, 98685529, 96825820, 78707346, 76954413, 44012174, 55619247, 12255439, 36693635, 64657547, 28549580, 81074679, 78165650, 56757562, 35979130, 43878306, 28194044, 73878508]

# work dir
config['work_dir'] = "./logs/benchmarks/nyu_v2_c/pspnet_resnet50/single_task_normal_estimation_run_1"
