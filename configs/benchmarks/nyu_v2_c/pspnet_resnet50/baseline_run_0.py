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
from configs.common.optimizers.baseline import optimizer_config
from configs.common.optimizers._core_ import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 40408308
config['train_seeds'] = [15268043, 78881063, 12670328, 63632629, 71591998, 65941860, 315210, 28535914, 670251, 17131411, 87471248, 85670141, 10907039, 32334913, 26098888, 41093679, 95706639, 87194050, 12955378, 80612092, 61107920, 48906560, 35906272, 45450918, 5922751, 81102149, 24912887, 50231544, 27860331, 20002907, 85642627, 70599038, 41480570, 6437850, 22937920, 70475863, 56937269, 26064528, 23509884, 52231368, 25112674, 8557932, 43605246, 2571946, 47951272, 46554899, 99948383, 33054315, 93959030, 26032281, 31084686, 56586270, 92471546, 81502031, 18721563, 52960760, 11608355, 71413350, 35873724, 10407995, 31689427, 16508969, 45053017, 31610893, 74236365, 35873822, 55953857, 27867116, 85775002, 37865225, 74133966, 40566196, 55640268, 21528914, 91322539, 91280286, 82624915, 23098487, 97899621, 69818091, 47432187, 15325369, 53523487, 37533441, 81913722, 37675273, 2816447, 68392527, 31375741, 23453398, 96582481, 51134825, 69967700, 91816966, 86196323, 48371771, 55263898, 64277446, 73507899, 43253685]

# work dir
config['work_dir'] = "./logs/benchmarks/nyu_v2_c/pspnet_resnet50/baseline_run_0"
