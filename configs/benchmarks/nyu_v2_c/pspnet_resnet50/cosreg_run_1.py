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
from configs.common.optimizers.cosreg import optimizer_config
from configs.common.optimizers._core_ import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 46680808
config['train_seeds'] = [59140735, 27963256, 3137381, 85038014, 21903172, 47068899, 47586542, 18885992, 98416800, 63370720, 14778160, 27822037, 96937188, 83527566, 88683171, 49411199, 91519913, 19993173, 27948118, 36344576, 51954771, 86658829, 30919410, 4278207, 9209330, 85435207, 18117210, 56234962, 31023257, 78941046, 62546769, 6638391, 25573533, 84722794, 28756627, 24871843, 46365983, 33411666, 59696232, 84070583, 43499761, 66543587, 46668520, 54417065, 90100966, 96118758, 26539529, 3290982, 88870371, 27851877, 58470352, 52862066, 9193370, 40001540, 39620567, 26929811, 44372038, 80708395, 13789051, 46436010, 94780900, 53694560, 34952299, 65940583, 75518856, 45011671, 84935544, 94195000, 7835550, 96264633, 53023990, 99436726, 46425078, 45956458, 82347659, 78068824, 26295160, 75940313, 80534989, 2525515, 23498778, 3330189, 95241635, 9020635, 29723959, 9295105, 36402134, 75340188, 46172677, 36108090, 16515931, 16853134, 24435961, 40664032, 21264280, 56978975, 9630235, 31235276, 23308540, 73752103]

# work dir
config['work_dir'] = "./logs/benchmarks/nyu_v2_c/pspnet_resnet50/cosreg_run_1"
