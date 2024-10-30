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
from configs.common.optimizers.rgw import optimizer_config
from configs.common.optimizers._core_ import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 82273195
config['train_seeds'] = [34080124, 68070568, 52980595, 56663777, 73745304, 12793068, 78380155, 44060094, 15192204, 35707313, 56775997, 38850177, 69257401, 56945353, 15458017, 64632012, 74928136, 736581, 51160240, 70549862, 73400526, 25041968, 21469912, 98582628, 11492434, 57505793, 15649202, 97488854, 25979432, 12959824, 80943611, 64077714, 46421945, 49595210, 29806938, 95196933, 85076368, 25210158, 24153947, 69678738, 73340779, 36950144, 36467603, 34839890, 51811243, 46080626, 29098893, 29618185, 18609971, 64216654, 66451235, 17899831, 7305734, 26358507, 53380260, 49063393, 94541092, 2303972, 90947390, 6067790, 71076534, 61557592, 93089604, 8652098, 38848972, 77523238, 34309655, 63146329, 82321456, 17997561, 50852792, 2339956, 83252664, 50680244, 96567409, 69891306, 49200734, 16076233, 2447141, 320402, 16155524, 5831746, 99263203, 26439792, 64730485, 16861169, 23562973, 44499399, 76248946, 2528839, 55405014, 64868857, 17012347, 89904021, 86737147, 21078084, 7398574, 40414193, 63383699, 98716564]

# work dir
config['work_dir'] = "./logs/benchmarks/city_scapes_c/pspnet_resnet50/rgw_run_0"