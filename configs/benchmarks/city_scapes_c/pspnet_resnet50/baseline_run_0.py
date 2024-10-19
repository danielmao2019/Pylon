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
config['init_seed'] = 85486362
config['train_seeds'] = [57314929, 67913984, 23802000, 73119197, 49948279, 72583235, 56059340, 31741871, 95361736, 3683306, 73122856, 93300148, 71339135, 82046941, 48718446, 80832160, 45143828, 5952313, 16945315, 94195546, 67811436, 34365249, 13431346, 22523181, 70254360, 19824232, 18706244, 48476942, 96754943, 32161233, 91334421, 45851221, 15295644, 59848560, 70234258, 56626129, 16748648, 9733194, 67777166, 87463855, 55033531, 30216429, 87620288, 53649605, 9130602, 14320719, 85974385, 69568427, 1261517, 56996909, 52061607, 78173401, 30765630, 47044768, 32822285, 70041016, 66284601, 33904360, 78205783, 88223065, 10157496, 22802351, 56096239, 84512944, 29422349, 120924, 15155184, 63664182, 56049961, 72372985, 34509665, 33098540, 74506391, 87283853, 84889068, 14830146, 92905164, 28552174, 57119441, 59736870, 4412250, 62548427, 86053272, 35794703, 62367713, 67207210, 90565810, 55863255, 56693542, 38538576, 65872489, 14480957, 53355058, 80393891, 40466171, 29252018, 11091938, 55455595, 97354163, 60209577]

# work dir
config['work_dir'] = "./logs/benchmarks/city_scapes_c/pspnet_resnet50/baseline_run_0"
