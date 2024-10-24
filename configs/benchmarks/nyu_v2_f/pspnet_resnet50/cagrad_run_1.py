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
from configs.common.optimizers.cagrad import optimizer_config
from configs.common.optimizers._core_ import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 37214281
config['train_seeds'] = [99922833, 11403272, 11588642, 96355976, 45926408, 16363522, 82177513, 85486740, 94873481, 52165681, 65542196, 23958753, 97569966, 78032487, 33201568, 14102765, 12779295, 88125056, 68936341, 11352695, 18431702, 83539073, 83417741, 89128353, 51982215, 34044505, 80038338, 77300016, 21383588, 74543330, 7581961, 55699474, 48760526, 47508906, 28451895, 16193676, 98848141, 74090785, 86254817, 2804306, 49549411, 63885720, 68913749, 79644338, 9239912, 18720101, 77896571, 92067335, 45518361, 37471733, 52661917, 55519166, 65585993, 35152904, 56634887, 64548403, 73850258, 70185972, 81580541, 48500782, 58356534, 84247879, 94087667, 22899527, 74408363, 17832675, 9986430, 84455543, 19273692, 34334270, 98477906, 4551581, 89837996, 49352854, 70049849, 7016415, 59817520, 725553, 31252527, 83285536, 27026338, 51173179, 28881741, 22093286, 97378569, 46094433, 46251204, 8964316, 40525371, 94104663, 58116701, 17370832, 92229180, 20901788, 52514583, 61387532, 85913276, 7207001, 54782152, 57493359]

# work dir
config['work_dir'] = "./logs/benchmarks/nyu_v2_f/pspnet_resnet50/cagrad_run_1"
