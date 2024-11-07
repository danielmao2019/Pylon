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
from configs.common.datasets.multi_mnist import config as dataset_config
config.update(dataset_config)

# model config
from configs.common.models.multi_mnist.multi_mnist_lenet5 import model_config_all_tasks as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.graddrop import optimizer_config
from configs.common.optimizers._core_ import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 7501715
config['train_seeds'] = [44495678, 15771281, 48620099, 43482524, 60677692, 21408043, 22253313, 65414726, 60663152, 19659528, 58081014, 82818777, 39286080, 90934340, 43341544, 99891214, 16133496, 91598, 37842478, 46711902, 96585138, 22323406, 96991326, 16826943, 63524812, 30117657, 45885707, 3950159, 75342541, 980559, 81964946, 75880154, 88066699, 80509338, 49969047, 42972356, 21023161, 49756201, 36287073, 11405407, 98636779, 25746100, 16708698, 72214802, 16424514, 90686739, 17092598, 68669166, 11240616, 36529703, 10703819, 66671801, 28215376, 42870660, 11165428, 49139974, 75678491, 20833501, 58142862, 4853413, 91016644, 34104674, 36777891, 83798938, 27157030, 96876568, 20931617, 38565300, 31283162, 91164281, 27314963, 18047009, 96266441, 25338623, 82005656, 63531246, 54539522, 29575014, 67943293, 72113777, 75464501, 82651506, 88200212, 8935434, 5163759, 8131403, 74792849, 38031573, 47689704, 667709, 68894155, 937039, 2549450, 74816802, 92593033, 87867114, 65400851, 64846299, 48666357, 74270105]

# work dir
config['work_dir'] = "./logs/benchmarks/multi_mnist/multi_mnist_lenet5/graddrop_run_1"