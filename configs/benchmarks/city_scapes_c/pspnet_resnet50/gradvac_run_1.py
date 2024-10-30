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
from configs.common.optimizers.gradvac import optimizer_config
from configs.common.optimizers._core_ import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 31688032
config['train_seeds'] = [10561203, 18474363, 34036605, 53184666, 74548003, 9972611, 65211652, 36737455, 89134028, 27331803, 17759537, 2109628, 85442038, 48129853, 17129283, 65010694, 72848932, 8383017, 68293271, 75613948, 40229323, 5764030, 99462512, 89919883, 86960080, 89761368, 94640543, 77406231, 93033404, 27484342, 75127534, 17778850, 25802005, 33708432, 60614208, 54648091, 48907881, 43732023, 74389161, 77055233, 6275647, 96916847, 70449427, 59436956, 58359331, 70620310, 97184702, 11710554, 73766921, 82882471, 4744705, 77753069, 80184690, 75127872, 9726141, 78113424, 52117542, 70894967, 106438, 4075302, 62431272, 18407293, 83087774, 7008943, 50896945, 762446, 19154681, 85320616, 20262462, 22661634, 37943437, 81778396, 70846421, 1933481, 18603685, 34993386, 25440635, 78777332, 30452616, 53896572, 9571917, 15659667, 53886254, 12580431, 83315818, 97809399, 37565203, 45723899, 74219460, 2078696, 30861679, 60957323, 74813142, 85773780, 77621775, 9436065, 67002555, 38748514, 44788872, 64871652]

# work dir
config['work_dir'] = "./logs/benchmarks/city_scapes_c/pspnet_resnet50/gradvac_run_1"