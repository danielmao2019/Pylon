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
from configs.common.optimizers.alignedmtl_ub import optimizer_config
from configs.common.optimizers._core_ import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 39423886
config['train_seeds'] = [67994664, 62040692, 81151047, 17711277, 55553907, 63613163, 26151948, 87218284, 25240326, 37105419, 11386908, 95370112, 64850445, 58522739, 16424491, 76727088, 19114695, 7611176, 28963308, 3089691, 58969708, 88212589, 62704202, 69597253, 3503203, 28329801, 66816237, 68896086, 27399778, 97895622, 39148952, 7806270, 81531662, 12412212, 88613602, 89838968, 95401898, 19872730, 57128807, 73907640, 47828583, 71138815, 11841048, 27702824, 46865037, 38000294, 32703718, 15949202, 55447683, 66120907, 64200345, 50403640, 8433771, 72889551, 39506030, 46746377, 44025162, 70185511, 66432737, 42545964, 21793656, 87838886, 48455158, 99789205, 36383443, 57400889, 47824459, 52337019, 68982403, 68592742, 90127492, 20902887, 86444984, 27786732, 41362454, 25177029, 59311655, 31548447, 24484148, 43620417, 63221394, 37439170, 59205723, 95121382, 86417379, 60154070, 11126409, 8162837, 85571526, 72926191, 82779533, 49338775, 51917206, 34736923, 95086668, 84765321, 91142018, 84059721, 97045, 87893590]

# work dir
config['work_dir'] = "./logs/benchmarks/nyu_v2_f/pspnet_resnet50/alignedmtl_ub_run_1"