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
from configs.common.datasets.city_scapes_f import config as dataset_config
config.update(dataset_config)

# model config
from configs.common.models.city_scapes_f.pspnet_resnet50 import model_config_all_tasks as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.cosreg import optimizer_config
from configs.common.optimizers._core_ import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 97277679
config['train_seeds'] = [85175355, 83407616, 89610270, 86888824, 63139740, 44860208, 63168328, 68432370, 61677874, 41070266, 60189732, 3129375, 62430196, 24825781, 84125559, 40618753, 22738529, 81680662, 32939490, 90929781, 2396651, 63781907, 14390094, 74836356, 95119833, 14848602, 63654385, 64203423, 26854739, 1576045, 81103587, 15823005, 87349787, 36275579, 21603831, 65556833, 14823538, 92429991, 12186914, 36219496, 96974527, 90817889, 40267890, 36441438, 79756301, 77868449, 6554856, 82431560, 4896924, 42013515, 56229383, 4452087, 37421646, 3173708, 31091324, 32735284, 72909770, 84447859, 43420155, 90943312, 32469944, 58099841, 35967391, 2278968, 88288318, 88113096, 89194740, 41007409, 61256958, 11176530, 23002662, 48379714, 95375761, 9253849, 60999888, 53886924, 1793624, 74723913, 44322131, 7651694, 80571715, 10150279, 95641873, 87818803, 29968127, 19071132, 20825239, 23984989, 7330666, 97664474, 49813496, 44057940, 18898927, 82400624, 13340831, 55058684, 33536140, 25959665, 92253216, 92533933]

# work dir
config['work_dir'] = "./logs/benchmarks/city_scapes_f/pspnet_resnet50/cosreg_run_2"
