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
from configs.common.datasets.multi_task_learning.city_scapes_f import config as dataset_config
config.update(dataset_config)

# model config
from configs.common.models.multi_task_learning.city_scapes_f.pspnet_resnet50 import model_config_all_tasks as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.multi_task_learning.pcgrad import optimizer_config
from configs.common.optimizers.standard import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 52543286
config['train_seeds'] = [63687875, 76723411, 99838262, 71046453, 73148934, 58480514, 49063292, 75081980, 51239478, 94319269, 68419409, 36877380, 58951863, 5389989, 80478663, 43129390, 9676074, 27392850, 10931025, 7338254, 31636327, 78143271, 34830435, 21810365, 89494949, 92363126, 77413328, 75063502, 67277924, 2806493, 40452670, 31855525, 70262716, 35795629, 28825413, 95730119, 50350040, 7479616, 42398495, 80757664, 83933387, 2648778, 38682307, 13491856, 22337399, 7886212, 12436584, 52208101, 1482936, 7200025, 41928563, 24539252, 53793414, 11549539, 95790653, 1468348, 47379652, 13923085, 35039771, 21000266, 35111605, 69582236, 45705227, 24496662, 71006022, 63963470, 26014375, 96665546, 71974374, 69230642, 14931966, 83596474, 31666350, 77977092, 74494938, 9113655, 97944729, 80906998, 23802020, 26931454, 53164876, 62497560, 26379597, 27237421, 65907797, 23015448, 25387304, 32571830, 40536722, 13258326, 14387798, 2983112, 31583912, 69270942, 62287089, 95988141, 60146629, 98472654, 36425666, 43897843]

# work dir
config['work_dir'] = "./logs/benchmarks/multi_task_learning/city_scapes_f/pspnet_resnet50/pcgrad_run_0"
