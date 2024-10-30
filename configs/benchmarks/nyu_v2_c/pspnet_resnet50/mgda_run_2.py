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
from configs.common.optimizers.mgda import optimizer_config
from configs.common.optimizers._core_ import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 37195344
config['train_seeds'] = [58055591, 78400442, 19090936, 23150845, 59140719, 54460846, 80443521, 4432211, 73440632, 47243738, 4613336, 11516267, 81419697, 94480883, 75429968, 78941707, 66248028, 89924237, 7595776, 96312565, 9044614, 63737804, 24403208, 6300270, 85243589, 49823791, 42432538, 27901674, 28534817, 42216800, 45588489, 8188922, 1760172, 78976633, 97600800, 85535781, 12360615, 73561845, 32742238, 43017779, 32159174, 52674952, 85865274, 14240916, 54469390, 68974269, 21618122, 74427908, 72220688, 35946098, 67007262, 66821973, 9678062, 24382031, 76394615, 50210944, 59596271, 7089480, 82041072, 58755079, 64539021, 72417284, 29732609, 77678904, 32869019, 69960466, 10147634, 21010200, 68682688, 36688110, 37893758, 97015129, 36744637, 54090364, 83421061, 55145742, 74724602, 21379775, 71329336, 81167286, 79966539, 60138391, 18826814, 71939394, 50263693, 16867939, 16447629, 2887068, 99800273, 87522708, 692444, 22979864, 74874129, 61819517, 26922743, 75534848, 93365247, 6056976, 11223205, 19416386]

# work dir
config['work_dir'] = "./logs/benchmarks/nyu_v2_c/pspnet_resnet50/mgda_run_2"