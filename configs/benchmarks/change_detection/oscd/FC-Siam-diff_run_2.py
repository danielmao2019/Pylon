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
from configs.common.datasets.change_detection.oscd import config as dataset_config
config.update(dataset_config)

# model config
from configs.common.models.change_detection.fc_siam import model_config
config['model'] = model_configconfig['model']['args']['arch'] = "FC-Siam-diff"

# optimizer config
from configs.common.optimizers.single_task_optimizer import single_task_optimizer_config
config['optimizer'] = single_task_optimizer_config

# seeds
config['init_seed'] = 85244322
config['train_seeds'] = [13587015, 34984111, 57801790, 90909375, 25430708, 36599267, 90377057, 27753989, 67928638, 14230174, 7601385, 60960479, 39666917, 30737282, 78781673, 39891904, 32731616, 33327442, 37851356, 42206856, 95820826, 28341275, 11370774, 29278353, 42097897, 83174340, 51159848, 34553805, 19927951, 73771226, 72395653, 72993020, 28368029, 45793713, 38875522, 91030728, 75552435, 90949444, 46503923, 21175472, 79503665, 91587714, 95319117, 59104208, 59475711, 90143719, 4999589, 70635088, 52058448, 80267797, 50173520, 90746152, 5296929, 25870261, 14897476, 14607482, 21008428, 99694288, 34613261, 48155600, 84482669, 92199821, 17256475, 34671581, 361536, 95869124, 25989743, 93807961, 20997923, 63887379, 37657966, 71448921, 68419502, 60300776, 70659842, 58645336, 44105648, 94948297, 26583561, 71251622, 24357711, 24488861, 15091688, 40134308, 70161907, 24987121, 63375107, 47300908, 52095144, 83568877, 35974864, 99563134, 73371012, 62336492, 60841002, 33625390, 95226450, 54915065, 68807287, 56796480]

# work dir
config['work_dir'] = "./logs/benchmarks/change_detection/oscd/FC-Siam-diff_run_2"
