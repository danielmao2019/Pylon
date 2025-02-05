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
from configs.common.datasets.multi_task_learning.nyu_v2_f import config as dataset_config
config.update(dataset_config)

# model config
from configs.common.models.multi_task_learning.nyu_v2_f.pspnet_resnet50 import model_config_all_tasks as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.multi_task_learning.rgw import optimizer_config
from configs.common.optimizers.standard import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 66799817
config['train_seeds'] = [25951593, 13839881, 99618590, 33191924, 94624490, 33135877, 22055379, 60512347, 72497778, 3308518, 93652457, 50205778, 23650734, 38205948, 23658147, 26790489, 26198614, 27473827, 97924218, 43305020, 55285270, 82546067, 56041445, 78366061, 8524107, 15658848, 5822008, 10262938, 12326746, 26088158, 99372661, 53255805, 42998041, 40878428, 68136858, 51216696, 55100567, 18307636, 25030573, 24710666, 70764608, 15476586, 24927149, 58111720, 84763744, 86105356, 64575962, 46229401, 26204084, 81426334, 84663818, 32018531, 20382245, 70606316, 82886575, 55179362, 46081027, 29674433, 27169082, 54386684, 39951035, 34157940, 51634043, 97455699, 96606197, 7263916, 42646568, 85426974, 10281488, 64303003, 52857038, 77314576, 91952040, 78587191, 29408423, 55792942, 1831477, 21648382, 64432683, 99364057, 21421257, 39732307, 75683601, 97537794, 2602925, 34110369, 78163420, 34336885, 86855753, 994567, 8900554, 26877230, 97102120, 16208332, 85097064, 8208074, 86568516, 88796858, 7068261, 30746659]

# work dir
config['work_dir'] = "./logs/benchmarks/multi_task_learning/nyu_v2_f/pspnet_resnet50/rgw_run_1"
