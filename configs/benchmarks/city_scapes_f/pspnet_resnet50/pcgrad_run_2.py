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

from runners import SupervisedMultiTaskTrainer
config['runner'] = SupervisedMultiTaskTrainer

# dataset config
from configs.common.datasets.city_scapes_f import config as dataset_config
config.update(dataset_config)

# model config
from configs.common.models.city_scapes_f.pspnet_resnet50 import model_config_all_tasks as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.pcgrad import optimizer_config
from configs.common.optimizers._core_ import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 681501
config['train_seeds'] = [46129433, 98912507, 32083024, 71724351, 38416025, 58658317, 64710783, 39178172, 69149949, 32713924, 87699563, 67544661, 75072104, 74971680, 79817262, 80205671, 99157305, 38808705, 57303611, 29538194, 40093688, 98619495, 10859056, 34995276, 43450688, 88259841, 79696959, 64851533, 35745137, 63333881, 85618803, 56916074, 76412566, 5508586, 95704766, 44526318, 31092423, 3195241, 37519710, 73491227, 7461853, 58124848, 32894884, 47365234, 48333448, 56896116, 28832475, 15484670, 82049736, 30931631, 77389830, 72987369, 5023838, 88001723, 72128538, 90907195, 66995621, 22122524, 46442283, 81368260, 28237076, 54497262, 15961307, 68698498, 40200783, 87638912, 27487396, 61329070, 64182256, 64079036, 81136543, 78875082, 19696560, 44145229, 55690287, 15468144, 6004681, 51174688, 37597692, 67238334, 43246203, 23045846, 83357512, 81763863, 21733559, 80072929, 96087331, 87633925, 31365965, 39831339, 35115850, 29806024, 49774865, 42832307, 20927576, 66507666, 63955610, 31309732, 1683591, 48150379]

# work dir
config['work_dir'] = "./logs/benchmarks/city_scapes_f/pspnet_resnet50/pcgrad_run_2"