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
from configs.common.datasets.nyu_v2_c import config as dataset_config
config.update(dataset_config)

# model config
from configs.common.models.nyu_v2_c.pspnet_resnet50 import model_config_all_tasks as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.alignedmtl_ub import optimizer_config
from configs.common.optimizers._core_ import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 35617500
config['train_seeds'] = [49811689, 88564222, 22300865, 28640195, 75163068, 78033975, 64059964, 60450517, 28475157, 56245990, 43903195, 74576197, 99130612, 86074676, 45507416, 41772475, 11341456, 66764757, 56410102, 67305556, 69921999, 56362942, 40688233, 66768117, 72212429, 85824267, 8151167, 38480403, 80926373, 92544522, 8268996, 49819195, 49560130, 37189081, 31535205, 30123650, 70426880, 41690812, 42437135, 61624893, 88036934, 36143286, 81193777, 93087476, 62920838, 1186233, 23972585, 58255007, 21800095, 9553920, 20509371, 15194103, 78936892, 75804384, 27170240, 78135786, 84093249, 40484472, 54983774, 36947260, 80461754, 42796716, 51269367, 98229396, 12058483, 57707247, 67129769, 55144500, 56945589, 43828661, 91652158, 40771612, 62291923, 6055885, 3691840, 67790537, 82033697, 15478606, 71092915, 66601711, 45922732, 5991971, 45917840, 84677782, 30885731, 53758807, 25968484, 5143140, 18847504, 8253225, 33577736, 85660118, 42328725, 94473684, 63068028, 8798873, 22978084, 28617192, 25060651, 58808999]

# work dir
config['work_dir'] = "./logs/benchmarks/nyu_v2_c/pspnet_resnet50/alignedmtl_ub_run_0"
