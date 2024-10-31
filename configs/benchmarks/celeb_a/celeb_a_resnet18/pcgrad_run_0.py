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
from configs.common.datasets.celeb_a import config as dataset_config
config.update(dataset_config)

# model config
from configs.common.models.celeb_a.celeb_a_resnet18 import model_config_all_tasks as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.pcgrad import optimizer_config
from configs.common.optimizers._core_ import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 21989986
config['train_seeds'] = [28829778, 98573986, 36365576, 52805192, 10464859, 57334006, 2259974, 45749890, 38389003, 66213051, 36800975, 52699465, 30023384, 38340172, 8696621, 72513028, 28381821, 45530140, 78940416, 80603436, 33614863, 35003996, 40329032, 59249951, 5722652, 46451026, 73241727, 92160581, 88417065, 46199740, 89478799, 85961998, 24429436, 28794699, 34281047, 24175676, 45131001, 24639721, 11943473, 53119716, 51723179, 1230007, 24846307, 66763179, 97364649, 6232089, 55702384, 56978856, 87699546, 96558340, 83754047, 53330440, 33987381, 16397453, 62446909, 31620023, 25002601, 72670472, 99865701, 29142671, 70955719, 7304929, 99657634, 42530117, 39658258, 2280992, 95930372, 80398054, 20596572, 39102517, 21864446, 22089017, 50778503, 41449592, 8244602, 48039698, 61143249, 34207250, 81721090, 75330305, 16637641, 72581084, 96262512, 88612743, 19081390, 33252766, 42697995, 92791732, 55451419, 84963399, 16594906, 92709395, 17455884, 3682671, 36980637, 64961431, 25381446, 8272673, 90866903, 99338443]

# work dir
config['work_dir'] = "./logs/benchmarks/celeb_a/celeb_a_resnet18/pcgrad_run_0"
