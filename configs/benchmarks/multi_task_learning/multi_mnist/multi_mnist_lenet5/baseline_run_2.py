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
config['init_seed'] = 22858115
config['train_seeds'] = [31871777, 1964211, 61002444, 1037040, 63268431, 91591750, 44854714, 12629230, 94326787, 67039472, 25356361, 69372913, 12436362, 70908146, 43854189, 97120897, 75371695, 22468371, 65562719, 96748510, 19214180, 31569827, 73775987, 88186870, 369678, 67350044, 21120850, 28378433, 15562609, 11374706, 88620850, 87830800, 32828744, 89132115, 30319846, 85584210, 69408409, 29773512, 51387657, 66650398, 11763702, 93101975, 62863656, 99044202, 44683951, 91737680, 65900506, 59850457, 65935139, 9693088, 56845672, 46144015, 50169833, 37718180, 49408727, 69766399, 86883516, 49311901, 20980876, 76523458, 93835005, 80886501, 49020415, 84358973, 96627952, 77501993, 30969520, 25012968, 37572112, 9002663, 99564100, 31483641, 56740933, 92157701, 44842058, 55589547, 87790230, 60520776, 5523528, 86419429, 97049894, 93907094, 81436409, 97236696, 29920034, 95761000, 7654123, 23311468, 37612749, 47487773, 90717589, 28498858, 39332062, 44104755, 72341552, 17504358, 20293390, 89565295, 69138444, 4876595]

# work dir
config['work_dir'] = "./logs/benchmarks/multi_task_learning/multi_mnist/multi_mnist_lenet5/baseline_run_2"
