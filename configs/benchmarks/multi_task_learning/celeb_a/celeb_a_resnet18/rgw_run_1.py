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
from configs.common.datasets.multi_task_learning.celeb_a import config as dataset_config
config.update(dataset_config)

# model config
from configs.common.models.multi_task_learning.celeb_a.celeb_a_resnet18 import model_config_all_tasks as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.multi_task_learning.rgw import optimizer_config
from configs.common.optimizers.standard import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 96310088
config['train_seeds'] = [15300818, 77462244, 17797033, 31048011, 92869201, 82117830, 12706848, 16293455, 90256970, 81437783, 16078068, 24812835, 65630036, 87603052, 46207853, 57254151, 92740310, 58067016, 36932324, 39903102, 77282518, 17779580, 96640017, 8106002, 26871451, 53161347, 8219650, 11998551, 84117977, 72855316, 19478988, 50846631, 21465874, 33710628, 21713654, 64842064, 12946645, 48749730, 40608478, 6610348, 22632458, 79164690, 61753789, 24878236, 89015723, 53701095, 17673523, 77691882, 78589224, 38075892, 67855416, 88843924, 35655906, 70162371, 8368516, 48151780, 60443833, 70093099, 86822368, 40254407, 25521627, 11171051, 67178949, 78954991, 96573120, 21042446, 46669145, 31421250, 96809447, 67580903, 17294695, 1627401, 4202039, 23266101, 96117464, 17717064, 91522893, 52586627, 33828255, 30712653, 98994154, 89903037, 81485259, 7018429, 51206124, 76666448, 66609164, 63542643, 43189064, 45757629, 76726310, 71917031, 24493329, 45541682, 84587987, 31391545, 51619304, 40785470, 37963644, 98233591]

# work dir
config['work_dir'] = "./logs/benchmarks/multi_task_learning/celeb_a/celeb_a_resnet18/rgw_run_1"
