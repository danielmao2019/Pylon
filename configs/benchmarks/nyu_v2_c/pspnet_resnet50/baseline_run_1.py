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
from configs.common.datasets.nyu_v2_c import config as dataset_config
config.update(dataset_config)

# model config
from configs.common.models.nyu_v2_c.pspnet_resnet50 import model_config_all_tasks as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.baseline import optimizer_config
from configs.common.optimizers._core_ import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 16036039
config['train_seeds'] = [91362406, 34271150, 44343697, 44773232, 73070452, 32145893, 48939272, 65544439, 54521110, 40246866, 52383139, 8108526, 86026278, 89181819, 17816878, 25273288, 45952717, 55684305, 12109369, 18361510, 36235422, 77243513, 13186018, 7539564, 77960137, 29162992, 14407844, 32940233, 60937747, 36159168, 5117152, 91164753, 27494522, 52395498, 86283288, 54641088, 84851591, 95901286, 79225773, 3522671, 4552776, 88061279, 7264633, 85825729, 27033748, 41831423, 38570482, 24454737, 83737631, 11535980, 3748137, 98683051, 83810550, 83068222, 37697809, 74247610, 45854332, 71346631, 77324227, 5511034, 51820412, 69599909, 57141874, 79312128, 80281923, 49514468, 97907181, 62699572, 33086054, 24757350, 48679808, 37759308, 8477133, 7562554, 52238290, 7979534, 18486154, 19022839, 58688786, 19967302, 63783617, 68715866, 35718779, 57248821, 71157060, 16459309, 78099241, 31112507, 14455288, 84922166, 90689273, 53669787, 27397742, 927924, 1042815, 41723544, 63419934, 3939228, 40872185, 52604620]

# work dir
config['work_dir'] = "./logs/benchmarks/nyu_v2_c/pspnet_resnet50/baseline_run_1"
