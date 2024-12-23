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
from configs.common.datasets.city_scapes_c import config as dataset_config
config.update(dataset_config)

# model config
from configs.common.models.city_scapes_c.pspnet_resnet50 import model_config_all_tasks as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.mgda_ub import optimizer_config
from configs.common.optimizers._core_ import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 10344341
config['train_seeds'] = [43064033, 78506054, 84584458, 30502359, 65550769, 86065044, 81401583, 5229762, 19160488, 36383979, 75288941, 19464286, 32249479, 20381146, 32671686, 89694823, 66889747, 21814807, 28408019, 80913826, 26819757, 65262609, 47952858, 8572985, 60833206, 86690195, 2697603, 89362893, 2009348, 53392518, 65724687, 68100044, 81399929, 96802955, 38125687, 31425270, 74389968, 72996707, 26352640, 54544052, 14684820, 99369816, 10664994, 37792956, 12749232, 55415871, 14077855, 27229878, 57850479, 84405100, 34518138, 35411911, 21955836, 62145899, 64215465, 209283, 87090968, 32467978, 69880070, 84525917, 6855025, 56385831, 76689591, 69028658, 90061840, 59045581, 2261643, 9806475, 12533729, 8529834, 70241303, 72256549, 44219705, 64972495, 83051980, 69871453, 86979687, 77306031, 42194601, 27722858, 63319390, 91667263, 40617236, 77743236, 6506871, 91424153, 68410173, 7040884, 58008462, 89210213, 81585945, 95306001, 2970361, 8840105, 3029689, 86489096, 23030375, 66848290, 93949771, 65758759]

# work dir
config['work_dir'] = "./logs/benchmarks/city_scapes_c/pspnet_resnet50/mgda_ub_run_0"
