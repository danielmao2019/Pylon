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
from configs.common.optimizers.baseline import optimizer_config
from configs.common.optimizers.standard import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 35510115
config['train_seeds'] = [61423673, 87164789, 10894905, 40607842, 79002465, 44093228, 75472714, 73112951, 33562324, 85163809, 37812169, 5665169, 18662678, 42182834, 36668289, 79093041, 56692947, 52736070, 19328537, 70866095, 78398794, 45086116, 73487557, 34580498, 44094418, 73962433, 136820, 85832582, 39561663, 3852147, 5160066, 13658382, 33846657, 97609289, 27462994, 68921473, 90288355, 58317524, 81928255, 87578737, 3591854, 41899843, 55828711, 47705206, 35202050, 14697190, 8043700, 49940770, 54553106, 22735879, 65700502, 32230556, 34254513, 57101556, 67828763, 753238, 43639038, 47633083, 2708015, 55719551, 8061633, 11583658, 1746265, 33982922, 84133381, 51222863, 47822524, 86963812, 42172611, 53204462, 57849407, 8172394, 63434306, 31414429, 13332282, 7404996, 41572573, 80097083, 32377076, 71946276, 97636072, 97185403, 29293578, 6084144, 63537900, 11088457, 73806751, 16956216, 39074843, 78274995, 90342499, 27505871, 45859374, 93398042, 57158713, 13037504, 72215926, 33919642, 11613383, 80113433]

# work dir
config['work_dir'] = "./logs/benchmarks/multi_mnist/multi_mnist_lenet5/baseline_run_1"
