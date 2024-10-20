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
from configs.common.datasets.city_scapes_f import config as dataset_config
config.update(dataset_config)

# model config
from configs.common.models.city_scapes_f.pspnet_resnet50 import model_config_all_tasks as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.rgw import optimizer_config
from configs.common.optimizers._core_ import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 66643191
config['train_seeds'] = [57323821, 95530698, 1339086, 98619678, 13243347, 31558028, 44306122, 58851302, 41343717, 29062588, 62941791, 75986551, 79161105, 68185321, 99018198, 35511147, 99371798, 80876431, 68406759, 27369426, 47911955, 27613930, 63514738, 22779135, 77608255, 77240770, 97444955, 52581478, 88138912, 20862604, 48956038, 83664647, 22334869, 70636030, 79499345, 74827080, 37130196, 15686870, 14006526, 31827615, 66401397, 15776953, 94817234, 61032460, 60870913, 31751727, 68852290, 73510039, 82503166, 12250754, 3767159, 31752167, 2308774, 12401571, 61987137, 98815945, 16822318, 23269413, 50911445, 71378054, 40125885, 45982008, 59594301, 83848251, 94464770, 72753929, 72807911, 31150986, 48396949, 36279490, 59611875, 71147750, 22512889, 73939431, 25879400, 74296595, 28091014, 98629676, 43977265, 36037364, 98871145, 19706906, 63242483, 93668295, 4996127, 69010501, 75116265, 64304627, 10454251, 70606004, 93366425, 39555665, 41265548, 11363773, 26892303, 2417799, 57105678, 14520600, 28055424, 71924982]

# work dir
config['work_dir'] = "./logs/benchmarks/city_scapes_f/pspnet_resnet50/rgw_run_1"
