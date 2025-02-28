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
from configs.common.optimizers.multi_task_learning.graddrop import optimizer_config
from configs.common.optimizers.standard import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 64941177
config['train_seeds'] = [77066278, 24347310, 15220024, 56294356, 89733325, 15640399, 11106396, 65087768, 58433588, 70613707, 83353146, 73052936, 14831076, 87765897, 51116823, 44115260, 48962594, 28038331, 81188556, 23590560, 42644271, 86750566, 32103229, 78385357, 44613169, 11338052, 31937619, 54553717, 80417562, 26627125, 18709655, 37251930, 3519341, 29513458, 10507, 64340953, 47195395, 70719812, 33349140, 65376923, 99558982, 28131196, 49324391, 89656312, 77986650, 60732046, 42216661, 67735163, 47822361, 18673605, 49311187, 6553096, 58868349, 6193071, 46401422, 27771251, 75726006, 43568787, 97770308, 72876623, 37238191, 83218128, 54782673, 77844886, 87621836, 99233625, 74549485, 39572029, 74013935, 92736822, 37895099, 9733148, 7550812, 73443470, 12663523, 85536555, 66212748, 2217526, 4545599, 13835989, 61127251, 61677862, 7834872, 41300570, 23236518, 24172373, 81520934, 62286590, 94998398, 17211334, 25304039, 58122046, 75508290, 52916767, 59918753, 12449612, 47713708, 42822265, 40522997, 68152852]

# work dir
config['work_dir'] = "./logs/benchmarks/multi_task_learning/nyu_v2_c/pspnet_resnet50/graddrop_run_2"
