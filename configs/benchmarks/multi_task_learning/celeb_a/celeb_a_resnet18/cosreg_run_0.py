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
from configs.common.datasets.multi_task_learning.celeb_a import config as dataset_config
config.update(dataset_config)

# model config
from configs.common.models.multi_task_learning.celeb_a.celeb_a_resnet18 import model_config_all_tasks as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.multi_task_learning.cosreg import optimizer_config
from configs.common.optimizers.standard import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 61790187
config['train_seeds'] = [87399746, 9408930, 23803829, 91466597, 48963548, 16214960, 2652238, 62077009, 53931844, 41898689, 36863636, 30345438, 45300446, 90879794, 63519824, 69607449, 91262761, 30913736, 54902031, 60825937, 71073629, 68635387, 40697516, 50279299, 33117759, 90028945, 67417641, 63983729, 19581603, 1075033, 45722362, 45995004, 12826784, 70924969, 83335807, 18014306, 98578244, 79452686, 71165400, 25353839, 61305152, 11702480, 97172539, 29257209, 57041216, 74615361, 13211861, 10304924, 64318410, 1402165, 52742436, 82091625, 58058840, 67443499, 70675678, 48229120, 20507756, 15248196, 15155812, 91561272, 90620481, 48017982, 65536733, 5375453, 4524188, 82247001, 24267334, 6961947, 80707425, 90706437, 98056564, 53025731, 63446550, 33799095, 82906490, 27419316, 75224821, 54810425, 80159538, 40247222, 46457816, 18552843, 16772736, 4787929, 4121431, 94352129, 39877763, 18538061, 77323708, 5652224, 2001049, 99378046, 28592212, 73263108, 10998978, 35334859, 84038639, 99303416, 96586121, 96780491]

# work dir
config['work_dir'] = "./logs/benchmarks/multi_task_learning/celeb_a/celeb_a_resnet18/cosreg_run_0"
