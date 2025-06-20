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
    'val_seeds': None,
    'test_seed': None,
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
                'class': schedulers.lr_lambdas.WarmupLambda,
                'args': {},
            },
        },
    },
}

from runners import SupervisedMultiTaskTrainer
config['runner'] = SupervisedMultiTaskTrainer

# dataset config
from configs.common.datasets.multi_task_learning.nyu_v2_f import config as dataset_config
config.update(dataset_config)

# model config
from configs.common.models.multi_task_learning.nyu_v2_f.pspnet_resnet50 import model_config_all_tasks as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.multi_task_learning.gradvac import optimizer_config
from configs.common.optimizers.standard import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 27905368
config['train_seeds'] = [37636913, 77306575, 69604208, 9249008, 69027018, 99281503, 82407409, 76645535, 19140376, 67495728, 31601029, 14413061, 63136071, 60965130, 95995565, 9628347, 62441733, 98460291, 13902778, 93813406, 22088143, 80504637, 7179449, 50646580, 83871516, 17893519, 54320414, 54720583, 78697023, 74268387, 94692169, 10626761, 94075617, 45285450, 5275542, 78573719, 49887740, 76837727, 30687486, 60123307, 40143481, 88380665, 74768827, 79379980, 66737274, 12773671, 2510116, 66139117, 16571015, 61433178, 10072516, 14840987, 35320941, 75719172, 7268862, 78042379, 93904425, 77216597, 59316789, 52202365, 99868343, 38334522, 75020239, 47551652, 62186961, 717877, 46768731, 26603561, 38511601, 97602166, 86682872, 36115701, 14846493, 25121568, 37116190, 21798199, 89322274, 18717585, 73203646, 40322742, 35315774, 81697370, 87131248, 7072870, 42380988, 4018368, 96153051, 96304050, 22415943, 7048120, 46508208, 66154316, 65946665, 20615951, 45673481, 98268110, 76723880, 43061460, 69478526, 69400027]
config['val_seeds'] = [5775146, 34693911, 83563413, 93196253, 54187107, 35206672, 99399924, 71844869, 62132165, 25579774, 60770446, 80131685, 16520715, 8231900, 51844004, 29296195, 70996747, 66118034, 63986977, 14413909, 43210968, 32448116, 13359134, 86422367, 3271925, 79798339, 67602759, 16590108, 3553080, 14503042, 33186048, 22484424, 51383810, 88909322, 65671180, 37981881, 58110564, 93976302, 52540460, 96661286, 33463618, 26608096, 95421459, 57143505, 32361692, 93032032, 31318086, 65490670, 49715773, 20825329, 19319778, 88094616, 46843549, 58095209, 29043539, 33058753, 45291231, 67088985, 72728923, 11005427, 48311040, 16350146, 17489406, 1071036, 24273181, 89441117, 27523737, 57782370, 68646845, 79511185, 66982790, 12824159, 87463833, 13182706, 67957112, 14703126, 41908021, 11556659, 76663335, 57996495, 85739252, 65530055, 79817018, 18856203, 57954505, 39007603, 2420679, 12719367, 44066459, 9370161, 36552252, 96571432, 21928141, 38115635, 2397482, 80504140, 82386469, 56537113, 57461981, 95569599]
config['test_seed'] = 84244405

# work dir
config['work_dir'] = "./logs/benchmarks/multi_task_learning/nyu_v2_f/pspnet_resnet50/gradvac_run_2"
