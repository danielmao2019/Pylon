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
from configs.common.datasets.multi_task_learning.city_scapes_f import config as dataset_config
config.update(dataset_config)

# model config
from configs.common.models.multi_task_learning.city_scapes_f.pspnet_resnet50 import model_config_all_tasks as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.multi_task_learning.graddrop import optimizer_config
from configs.common.optimizers.standard import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 91816632
config['train_seeds'] = [81676093, 11737856, 78539046, 18074488, 18414437, 43565928, 22502178, 92126734, 33451410, 86023363, 5553854, 16640218, 15610151, 16446987, 96427638, 9215110, 56870485, 61956612, 83936106, 44968304, 38956803, 84256523, 70655161, 68567409, 30127378, 98251886, 81179959, 8093446, 57002287, 35077069, 99618665, 66166746, 21043229, 19992012, 44274436, 86062813, 54964247, 82556813, 34125043, 32857532, 6465822, 14413696, 2615739, 35570353, 91147307, 96528600, 21963156, 52089785, 28566389, 66719500, 83638011, 23386938, 64598149, 469281, 19457917, 51807544, 79598030, 14842945, 96230245, 43322202, 72038601, 45491152, 66400846, 66830939, 16020891, 45184209, 85958436, 79147636, 1183624, 90250814, 43181320, 62054639, 1504631, 32010795, 51704024, 53327234, 21144830, 19474043, 35455670, 83628893, 17544942, 35496982, 90146177, 63078550, 37971509, 45928144, 92549719, 14255655, 64211555, 85651324, 77648936, 84758032, 77499866, 59548411, 71526896, 81363319, 61327922, 69281948, 48505150, 28350934]

# work dir
config['work_dir'] = "./logs/benchmarks/multi_task_learning/city_scapes_f/pspnet_resnet50/graddrop_run_2"
