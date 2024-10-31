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
from configs.common.datasets.city_scapes_c import config as dataset_config
config.update(dataset_config)

# model config
from configs.common.models.city_scapes_c.pspnet_resnet50 import model_config_all_tasks as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.baseline import optimizer_config
from configs.common.optimizers._core_ import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 85903523
config['train_seeds'] = [60308474, 5520482, 15796293, 19451305, 91544661, 52248237, 68646774, 81909701, 96261691, 78809332, 38489596, 42134779, 13292434, 29698212, 19682372, 89789225, 77919298, 18633571, 84624079, 85245258, 97373580, 21992173, 90931295, 97544179, 84463774, 2694523, 93058427, 14022754, 46033943, 85481317, 4776724, 65018500, 97984345, 17936556, 29533911, 44108604, 18580049, 40715704, 56768339, 60713420, 11180187, 33173266, 22355531, 75792208, 75271065, 42425090, 66331522, 76889345, 70633025, 85720556, 93060381, 75374931, 17840723, 48637836, 69972706, 63348197, 8230116, 54844890, 92970704, 6454967, 90016940, 16650345, 44341342, 16123816, 29154976, 14986854, 39636524, 13563444, 42864771, 53081792, 84715515, 59050555, 80114951, 6988158, 34150299, 47254931, 98663010, 10484574, 92135264, 2042717, 1212063, 69163717, 66040066, 26761932, 26667932, 33597612, 36969056, 85248396, 75993166, 23727892, 91414952, 62909339, 25496592, 79429584, 41557341, 36346246, 56811768, 17077756, 38342072, 86871830]

# work dir
config['work_dir'] = "./logs/benchmarks/city_scapes_c/pspnet_resnet50/baseline_run_1"
