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
from configs.common.optimizers.mgda import optimizer_config
from configs.common.optimizers._core_ import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 97483750
config['train_seeds'] = [87455983, 3217023, 10973003, 52652874, 51051314, 66389386, 99362929, 90069946, 51050705, 32211804, 33794125, 54132374, 22229943, 27495807, 92227139, 58621738, 6313465, 96202396, 83056134, 34681682, 9642289, 83378078, 34195907, 11352431, 52917796, 70247064, 10814870, 20609728, 70041930, 30391421, 19137508, 88015346, 79540816, 73805307, 15756490, 7766260, 38171716, 73089177, 81006514, 27764795, 73204363, 790113, 97695617, 86851462, 68289962, 23428178, 6011350, 70853673, 66795456, 34223325, 6060337, 65659821, 75762494, 37432904, 37221295, 45046992, 59245314, 16337328, 20231176, 73499678, 84163279, 8761786, 5674109, 40365085, 40257063, 28357197, 20281138, 3510177, 42562681, 34342680, 47888934, 77227154, 54017211, 74024589, 68019335, 94033348, 13945951, 16817905, 69887711, 3177062, 22014337, 97419189, 42247098, 30433009, 60942536, 2758035, 61150111, 60659967, 20078557, 38189221, 47814682, 97831452, 84069020, 70026365, 82512191, 17204829, 83429064, 36897595, 7847528, 36255877]

# work dir
config['work_dir'] = "./logs/benchmarks/nyu_v2_f/pspnet_resnet50/mgda_run_0"
