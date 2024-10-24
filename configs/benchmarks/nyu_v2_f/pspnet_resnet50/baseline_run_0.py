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
from configs.common.optimizers.baseline import optimizer_config
from configs.common.optimizers._core_ import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 93816562
config['train_seeds'] = [41858926, 85613281, 79674789, 37106053, 39177131, 5092339, 96077248, 25527468, 2374317, 94428317, 48663823, 61619588, 64789532, 39777332, 94067569, 81443784, 25546757, 84395029, 67773065, 26735138, 10911830, 28007458, 26814460, 757229, 54567325, 6281501, 40609197, 31899832, 62020039, 80968180, 97287039, 33665862, 27263514, 33339236, 77732759, 78201525, 99698593, 38330912, 75846543, 4505227, 93319892, 12263682, 15277514, 256979, 80243023, 88996380, 6908053, 4350796, 97062495, 97266944, 85453112, 75668015, 48314303, 70936139, 63030753, 70910985, 93253901, 32624430, 16253684, 695450, 30484212, 80771558, 31039486, 40366380, 35663319, 40044582, 25816657, 56693961, 5267057, 39763088, 53493692, 20126639, 33584472, 54443423, 52560429, 49744960, 76829548, 41738142, 18721409, 43431654, 9713813, 94126917, 77666264, 12491696, 57273703, 15292017, 96802212, 36546323, 63274248, 86316699, 98850849, 98262440, 50238607, 70416708, 42479340, 8754896, 30850774, 44599188, 57529872, 52135982]

# work dir
config['work_dir'] = "./logs/benchmarks/nyu_v2_f/pspnet_resnet50/baseline_run_0"
