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
from configs.common.optimizers.baseline import optimizer_config
from configs.common.optimizers._core_ import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 22241027
config['train_seeds'] = [75179828, 10158573, 13545435, 60457336, 50173028, 5249575, 37885601, 81322795, 28479924, 52495274, 94789762, 5595222, 22491683, 98105058, 32348742, 26153786, 17983217, 55815162, 72854057, 1780669, 80742609, 3238617, 83788827, 49245908, 59658615, 11204331, 30625157, 48272862, 23675772, 80827623, 25864474, 73029406, 74034378, 57072277, 11252525, 42049384, 24532847, 7924619, 79630962, 69254558, 59676000, 12410559, 69268261, 3407109, 68511381, 42441184, 66831066, 5087973, 93474227, 19058573, 94363056, 7792632, 17669313, 69875733, 34741399, 35644725, 87115582, 1192054, 57364960, 72697928, 81910669, 41475113, 36895094, 42936645, 68665013, 25852139, 29862229, 46252888, 43832335, 12822756, 46636658, 68822334, 53954111, 97536988, 93055371, 63246921, 90538000, 37867988, 99124535, 16292238, 49690469, 42531580, 55822410, 52740162, 94116264, 5078227, 90201824, 58597850, 42452114, 45024162, 85700476, 94109936, 67177939, 5982323, 48394669, 32853021, 2705762, 54392411, 54478018, 90346003]

# work dir
config['work_dir'] = "./logs/benchmarks/nyu_v2_f/pspnet_resnet50/baseline_run_1"
