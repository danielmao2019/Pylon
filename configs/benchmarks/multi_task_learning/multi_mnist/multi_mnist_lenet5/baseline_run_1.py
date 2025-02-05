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
from configs.common.datasets.multi_task_learning.multi_mnist import config as dataset_config
config.update(dataset_config)

# model config
from configs.common.models.multi_task_learning.multi_mnist.multi_mnist_lenet5 import model_config_all_tasks as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.multi_task_learning.baseline import optimizer_config
from configs.common.optimizers.standard import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 36091425
config['train_seeds'] = [65632085, 90270131, 12364828, 12729494, 53424368, 18031976, 82338096, 32913917, 44497635, 61737692, 81704871, 96436399, 59868620, 53895358, 97563911, 67866470, 3403944, 98643644, 63864613, 30602380, 45752012, 58750910, 76391136, 9387981, 21139302, 70429809, 91750528, 54051767, 54727474, 52469622, 72520528, 74307613, 82268604, 22339064, 80611616, 11066151, 6783707, 79440274, 28560745, 44716534, 87198911, 55247844, 8103873, 17510109, 67559597, 52361222, 14872767, 67270898, 22606718, 47209032, 98870549, 63040689, 75484146, 37067244, 56025361, 68900832, 94649326, 45372209, 32186818, 92350341, 13290219, 50106797, 68542422, 74619417, 34964881, 55385955, 29564541, 18253287, 17239379, 29183436, 96062672, 19883660, 67331811, 39782930, 64454008, 85867011, 49651547, 72187408, 21733285, 36662536, 83998837, 76392662, 8275526, 82214175, 95333968, 55605601, 74651194, 79306753, 3896395, 90912757, 7161630, 94272252, 83222279, 72169535, 95997320, 94728707, 85077824, 96543239, 63336437, 35400857]

# work dir
config['work_dir'] = "./logs/benchmarks/multi_task_learning/multi_mnist/multi_mnist_lenet5/baseline_run_1"
