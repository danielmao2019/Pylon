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

from runners import SupervisedMultiTaskTrainer
config['runner'] = SupervisedMultiTaskTrainer

# dataset config
from configs.common.datasets.multi_task_learning.city_scapes_c import config as dataset_config
config.update(dataset_config)

# model config
from configs.common.models.multi_task_learning.city_scapes_c.pspnet_resnet50 import model_config_all_tasks as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.multi_task_learning.mgda_ub import optimizer_config
from configs.common.optimizers.standard import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 98130582
config['train_seeds'] = [80676008, 87600308, 5235616, 88294793, 383742, 38429340, 93188597, 93738389, 76870027, 39580570, 73278204, 84975436, 89375757, 11162420, 59595371, 76880614, 33253660, 60281699, 72554289, 55587783, 41913682, 51564110, 10646363, 56996693, 67362320, 82741268, 75765732, 23262163, 91485043, 75773363, 47073015, 81137271, 87076227, 95223432, 7524544, 12916158, 92492228, 33500935, 51078711, 89935016, 86929992, 24251897, 41482912, 90567708, 33844116, 59928410, 75468210, 15753316, 30642157, 40438986, 1309158, 63157715, 2307848, 31318934, 10120431, 4006135, 51139929, 51892780, 44328585, 46096887, 69632670, 37648878, 73044336, 76522372, 71787965, 90968250, 28536308, 29389588, 70998512, 4228155, 4069969, 80723695, 63776046, 66055941, 63511183, 96379171, 94717124, 477697, 45751449, 65229374, 12417998, 14919742, 49054930, 34725048, 55792424, 41377377, 1421651, 67905266, 64825390, 87846488, 36543813, 23601381, 19969461, 79256004, 33739351, 61106575, 43112472, 32656634, 44492903, 68917391]

# work dir
config['work_dir'] = "./logs/benchmarks/multi_task_learning/city_scapes_c/pspnet_resnet50/mgda_ub_run_0"
