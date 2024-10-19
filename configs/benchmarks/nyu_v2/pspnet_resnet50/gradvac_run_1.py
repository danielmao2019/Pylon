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
from configs.common.datasets.nyu_v2 import config as dataset_config
config.update(dataset_config)

# model config
from configs.common.models.nyu_v2.pspnet_resnet50 import model_config_all_tasks as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.gradvac import optimizer_config
from configs.common.optimizers._core_ import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 75703810
config['train_seeds'] = [16291773, 12150921, 96317890, 30573523, 74398884, 41171624, 54266569, 5937807, 49097980, 1596917, 58448114, 50020229, 34156189, 96432922, 27226889, 28309555, 19729470, 91923839, 78380926, 62051618, 34766735, 6455915, 81793627, 85200233, 66674153, 5953450, 40564597, 76343009, 23804554, 86989457, 17219032, 94754807, 6675029, 66361721, 99538720, 47442927, 20708412, 17839045, 9263921, 66931935, 51355855, 88419777, 24469555, 39590944, 7814744, 62384086, 61820860, 44919645, 39577848, 10890838, 69636947, 54310277, 71211882, 57843121, 29299052, 5238951, 26692809, 12582580, 9363510, 31462238, 22639013, 91560704, 24753670, 47478497, 46373847, 67826272, 48412634, 82488261, 71110589, 14266569, 75629570, 37783351, 2915362, 24316230, 79872257, 85617517, 37742775, 32839637, 52992157, 18389260, 76741715, 55446948, 140719, 65389677, 98607090, 52420433, 10074204, 40240172, 95361809, 92091663, 65161343, 13124405, 20255622, 90312167, 22998262, 81482774, 19401821, 57931511, 62235473, 4763270]

# work dir
config['work_dir'] = "./logs/benchmarks/nyu_v2/pspnet_resnet50/gradvac_run_1"
