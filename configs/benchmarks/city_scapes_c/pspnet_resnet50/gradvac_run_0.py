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
from configs.common.optimizers.gradvac import optimizer_config
from configs.common.optimizers._core_ import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 39408440
config['train_seeds'] = [44833052, 36130995, 98212146, 80156252, 23894678, 99705309, 9376122, 74971157, 443847, 42564283, 80556840, 17998629, 26505115, 73969059, 59664373, 96307379, 23536587, 20088786, 91710731, 27674920, 10210381, 1280961, 81746517, 22930764, 98476326, 88695124, 70390345, 37510927, 20860149, 23885323, 20181477, 83004958, 45657981, 97200984, 9362903, 57376724, 22812803, 5971416, 41607667, 50901196, 38367988, 6246888, 35351754, 73392943, 50432022, 7778698, 476424, 73506154, 41005438, 84500688, 20169856, 97656721, 97377613, 21721211, 63288550, 69041160, 65713031, 43569316, 57253753, 65082419, 20005415, 94691268, 31466851, 49604458, 16439983, 86718467, 94904326, 70850724, 32931602, 69856761, 1241900, 11356557, 92713644, 50731594, 55619435, 98053949, 46094490, 56445293, 6518353, 86252098, 32778753, 70416676, 49140370, 7601386, 28046563, 98764392, 97208824, 52233912, 19337661, 21182514, 65671356, 86675343, 87396867, 3370128, 80636157, 59479733, 71762327, 23352331, 73127481, 23046541]

# work dir
config['work_dir'] = "./logs/benchmarks/city_scapes_c/pspnet_resnet50/gradvac_run_0"
