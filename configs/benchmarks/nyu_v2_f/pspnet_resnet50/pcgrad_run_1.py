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
from configs.common.datasets.nyu_v2_f import config as dataset_config
config.update(dataset_config)

# model config
from configs.common.models.nyu_v2_f.pspnet_resnet50 import model_config_all_tasks as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.pcgrad import optimizer_config
from configs.common.optimizers._core_ import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 65791888
config['train_seeds'] = [94777141, 48267407, 41616907, 61973064, 49287521, 62399014, 64103365, 89976649, 15881092, 27513308, 1993241, 31832104, 4602133, 52960594, 72407818, 91602826, 20649099, 5100353, 32710661, 89311529, 16282302, 99437596, 981784, 64290319, 1014521, 93034957, 28591843, 50084717, 84108046, 47125891, 95670395, 70073897, 73125214, 84145356, 35158707, 55940624, 58274820, 20486613, 95156457, 31829777, 59459038, 32431379, 55658906, 71595965, 54747801, 20828753, 20725302, 27534623, 63692508, 66532801, 38756826, 33636313, 89937297, 42137353, 20157679, 17157018, 99535076, 96629323, 36145561, 84796509, 79842338, 98451426, 93058044, 37774254, 31381151, 6155265, 68442075, 17677644, 29326059, 47006731, 18115979, 62737515, 70272123, 44396567, 90061733, 76181794, 43082149, 15537138, 15314715, 52865661, 7688496, 5858823, 50071059, 68488908, 82410300, 21924434, 54475822, 85317530, 35295047, 21255492, 40274995, 13226825, 28790095, 64553992, 53468874, 57416793, 37770691, 21986608, 27447909, 97252747]

# work dir
config['work_dir'] = "./logs/benchmarks/nyu_v2_f/pspnet_resnet50/pcgrad_run_1"
