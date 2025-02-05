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
from configs.common.datasets.multi_task_learning.nyu_v2_c import config as dataset_config
config.update(dataset_config)

# model config
from configs.common.models.multi_task_learning.nyu_v2_c.pspnet_resnet50 import model_config_all_tasks as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.multi_task_learning.gradvac import optimizer_config
from configs.common.optimizers.standard import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 84545678
config['train_seeds'] = [41058932, 89391293, 31245759, 53792825, 11768545, 92597919, 67796511, 46962497, 30791703, 5976765, 11102482, 98803946, 74776656, 75090781, 29866767, 80521324, 98449565, 38897994, 200822, 14360167, 76263552, 72367946, 39797507, 2901187, 96126701, 9831277, 38065757, 64923821, 24604650, 49058616, 89180092, 72546725, 18186871, 29515815, 53366635, 12238120, 3759501, 38439096, 99436649, 99197544, 30951521, 73158042, 62639862, 26953567, 67200540, 38886493, 15576017, 75713303, 93040450, 19548420, 61702970, 31283112, 62421304, 22092884, 93729812, 21836796, 30880884, 63568019, 58807255, 18848842, 13825128, 37367691, 77602755, 49927037, 2179516, 28773555, 92209633, 54861805, 86384700, 3945589, 65527675, 67169508, 12171759, 1358884, 16727661, 60995278, 17110318, 8149994, 47734481, 75626697, 77340022, 80801354, 5925011, 16853798, 59522910, 22494176, 14964264, 91001266, 92211344, 19070949, 30636056, 56833085, 42741727, 87586111, 38374010, 33064563, 19686640, 73033679, 2085487, 48363377]

# work dir
config['work_dir'] = "./logs/benchmarks/multi_task_learning/nyu_v2_c/pspnet_resnet50/gradvac_run_0"
