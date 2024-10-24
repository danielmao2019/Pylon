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
from configs.common.optimizers.mgda_ub import optimizer_config
from configs.common.optimizers._core_ import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 31850315
config['train_seeds'] = [34831077, 18824633, 65541908, 4871951, 36666921, 84980887, 86132694, 49944314, 32213965, 26318052, 26679260, 54421629, 27038753, 11896045, 78557651, 59256668, 87164150, 97358768, 68622654, 19231488, 15082603, 1580676, 5118583, 44660897, 67565689, 65837748, 51312790, 75928577, 54357521, 65291745, 16662831, 58005953, 11122975, 17755133, 65445160, 59600372, 54658509, 93784974, 87677394, 92741557, 565283, 43736133, 3571988, 1688427, 51173378, 76037479, 4319140, 41291269, 93189424, 68235370, 64788418, 70625101, 99565264, 70801544, 22306632, 38426529, 12212123, 28546291, 42449664, 41213320, 58099582, 97936345, 46661524, 85456671, 35146982, 43254488, 9797972, 81393638, 90328064, 24826259, 72781355, 74510384, 76246811, 31692047, 65559288, 32013871, 46550752, 23237476, 50029320, 49510498, 3326830, 77655540, 52832474, 29560187, 28152148, 87248474, 59761313, 64357460, 87439291, 5769149, 53834961, 18053574, 24590914, 46392704, 89446247, 70280831, 3897002, 26608515, 94261305, 93507125]

# work dir
config['work_dir'] = "./logs/benchmarks/nyu_v2_f/pspnet_resnet50/mgda_ub_run_0"
